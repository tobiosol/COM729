import os
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import fundus_v2.proj_util as proj_util

class SVMFeatureClassifier:
    def __init__(self, cnn_models, train_loader, val_loader, device):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.svm_classifier = make_pipeline(StandardScaler(), SVC(probability=True))
        self.feature_name = "_".join(model.model_name for model in cnn_models)
        self.feature_file = proj_util.get_trained_model(f"{self.feature_name}_svm_features.npy")
        self.label_file = proj_util.get_trained_model(f"{self.feature_name}_svm_labels.npy")
        self.svm_file = proj_util.get_trained_model("svm_model.pth")
        self.cnn_models = cnn_models
        self.loaded_svm_classifier = None
        self.scaler = None

    def load_cnn_models(self):
        for model in self.cnn_models:
            model_path = proj_util.get_trained_model(f"{model.model_name}_model.pth")
            loaded_model = torch.jit.load(model_path)
            model.model = loaded_model.to(self.device)
            model.model.eval()
            print(f'Model loaded from {model_path}')
        self.models = [model.model for model in self.cnn_models]

    @torch.no_grad()
    def extract_features(self, dataloader):
        all_features, all_labels = [], []
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            model_features = [model(inputs).cpu().numpy() for model in self.models]
            combined_features = np.concatenate(model_features, axis=1)
            all_features.append(combined_features)
            all_labels.append(targets.cpu().numpy())
        return np.concatenate(all_features), np.concatenate(all_labels)

    def save_features(self, features, labels):
        np.save(self.feature_file, {'features': features, 'labels': labels})
        print(f"Features saved to {self.feature_file}")

    def load_features(self):
        if os.path.exists(self.feature_file):
            data = np.load(self.feature_file, allow_pickle=True).item()
            return data['features'], data['labels']
        print(f"Feature file {self.feature_file} not found.")
        return None, None

    def evaluate(self):
        self.load_cnn_models()
        features, labels = self.extract_features(self.val_loader)
        features = TSNE(n_components=3, init='pca', learning_rate='auto').fit_transform(features)
        predictions = self.svm_classifier.predict(features)
        self.print_metrics(labels, predictions, features)

    def train(self):
        self.load_cnn_models()
        features, labels = self.load_features()
        if features is None or labels is None:
            features, labels = self.extract_features(self.train_loader)
            self.save_features(features, labels)

        features = TSNE(n_components=3, init='pca', learning_rate='auto').fit_transform(features)
        self.n_features = features.shape[1]
        
        param_grid = {
            'svc__C': [0.1, 1, 10],
            'svc__kernel': ['rbf', 'poly'],
            'svc__gamma': ['scale', 'auto']
        }
        
        grid_search = GridSearchCV(self.svm_classifier, param_grid, cv=StratifiedKFold(n_splits=5), n_jobs=1, verbose=3)
        grid_search.fit(features, labels)

        self.svm_classifier = grid_search.best_estimator_
        torch.save({'model': self.svm_classifier, 'n_features': self.n_features}, self.svm_file)
        print(f"Model saved to {self.svm_file}")


    def load_svm_classifier(self):
        loaded_data = torch.load(self.svm_file)
        self.loaded_svm_classifier = loaded_data['model']
        self.scaler = self.loaded_svm_classifier.named_steps['standardscaler']

    def _extract_image_features(self, image_tensor):
        print(image_tensor.min(), image_tensor.max())
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.from_numpy(image_tensor).float()
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            features = []
            for model in self.cnn_models:
                feature = model.extract_features(image_tensor)
                print("feature ", feature)
                features.append(feature)
            # combined_features = torch.cat(features, dim=1).squeeze().cpu().numpy()
            combined_features = torch.stack(features, dim=0).mean(dim=0).squeeze().cpu()

        combined_features = combined_features.reshape(1, -1)
        print("combined_features: ", combined_features)
        print(f"Number of features StandardScaler is expecting: {self.scaler.n_features_in_}")
        return combined_features
    
    def predict(self, image_tensor):
        self.load_cnn_models()
        if self.loaded_svm_classifier is None:
            self.load_svm_classifier()
        
        extracted_features = self._extract_image_features(image_tensor)
        standardized_features = self.scaler.transform(extracted_features)
        
        print(f"Number of standardized_features: {standardized_features}")
        
        prediction = self.loaded_svm_classifier.predict(standardized_features)
        prediction_probs = self.loaded_svm_classifier.predict_proba(standardized_features)
        return prediction.item(), prediction_probs[0]

    def print_metrics(self, labels, predictions, features):
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        print("Classification Report:\n", classification_report(labels, predictions))
        print("Classification Matrix:\n", cm)

        n_classes = len(np.unique(labels))
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels, self.svm_classifier.predict_proba(features)[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 7))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (area = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        overall_roc_auc = roc_auc_score(labels, self.svm_classifier.predict_proba(features), average='macro', multi_class='ovr')
        print(f"Overall ROC AUC: {overall_roc_auc:.4f}")
