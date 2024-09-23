from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_auc_score, roc_curve
import torch
import seaborn as sns
from fundus_v2.svm_classifier import SVMFeatureClassifier
from fundus_v2.random_forest_classifier import RandomForestFeatureClassifier
import fundus_v2.proj_util as proj_util

class EnsembleClassifier:
    def __init__(self, cnn_models, train_loader, val_loader, device):
        self.device = device
        self.svm_classifier = SVMFeatureClassifier(cnn_models, train_loader, val_loader, device)
        self.rf_classifier = RandomForestFeatureClassifier(cnn_models, train_loader, val_loader, device)
        self.ensemble_file = proj_util.get_trained_model("ensemble_model.pth")

    def train(self):
        self.svm_classifier.train()
        self.rf_classifier.train()
        torch.save({
            'svm_model': self.svm_classifier.svm_classifier,
            'rf_model': self.rf_classifier.rf_classifier
        }, self.ensemble_file)
        print(f"Ensemble model saved to {self.ensemble_file}")

    def predict(self, image_tensor):
        svm_prediction = self.svm_classifier.predict(image_tensor)
        rf_prediction = self.rf_classifier.predict(image_tensor)
        
        # Simple majority voting
        ensemble_prediction = np.round(np.mean([svm_prediction, rf_prediction], axis=0)).astype(int)
        
        return ensemble_prediction

    def load_model(self):
        loaded_data = torch.load(self.ensemble_file)
        self.svm_classifier.svm_classifier = loaded_data['svm_model']
        self.rf_classifier.rf_classifier = loaded_data['rf_model']
        print(f"Ensemble model loaded from {self.ensemble_file}")

    def evaluate(self, test_loader):
        self.load_model()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            predictions = self.predict(inputs)
            probabilities = self.predict_proba(inputs)
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
            all_labels.extend(labels.cpu().numpy())
        
        self.print_metrics(np.array(all_labels), np.array(all_predictions), np.array(all_probabilities))
        
    def predict_proba(self, image_tensor):
        svm_proba = self.svm_classifier.svm_classifier.predict_proba(self.svm_classifier.extract_features(image_tensor))
        rf_proba = self.rf_classifier.rf_classifier.predict_proba(self.rf_classifier.extract_features(image_tensor))
        
        # Average the probabilities from both classifiers
        ensemble_proba = (svm_proba + rf_proba) / 2
        
        return ensemble_proba
    
    def print_metrics(self, labels, predictions, probabilities):
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
            fpr[i], tpr[i], _ = roc_curve(labels, probabilities[:, i], pos_label=i)
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

        overall_roc_auc = roc_auc_score(labels, probabilities, average='macro', multi_class='ovr')
        print(f"Overall ROC AUC: {overall_roc_auc:.4f}")