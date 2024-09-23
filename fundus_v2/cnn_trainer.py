import os
import time

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = Path(current_dir) / 'fundus_v2'
sys.path.append(str(subdirectory_path))

from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from early_stopping import EarlyStopping
from fundus_v2 import cnn_models, image_dataset, image_transforms, proj_util

class FundusCNNTrainer:
    def __init__(self, model_name: str, train_loader: DataLoader, val_loader: DataLoader):
        self.model_name = model_name
        self.model_path = proj_util.get_trained_model(f"{model_name}_model.pth")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.early_stopper = EarlyStopping(patience=10)
        self.best_accuracy = 0
        self.learning_rates: List[float] = []

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self._initialize_training()
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print("No pre-trained model found. Train a new model.")

    def train(self, num_epochs: int) -> None:
        if os.path.exists(self.model_path):
            print("Model already trained. Skipping training.")
            self.load_model()
            return

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss, train_acc, train_metrics = self.train_one_epoch(self.train_loader)
            val_loss, val_acc, val_metrics = self.validate(self.val_loader)
            self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            self._print_epoch_results(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, val_metrics, current_lr, time.time() - start_time)

            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_model()

            if self.early_stopper(val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break

        self.plot_learning_rates()

    def train_one_epoch(self, train_loader: DataLoader, num_classes: int = 3) -> Tuple[float, float, Dict]:
        class_weights = self._compute_class_weights(train_loader, num_classes)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels.view(-1)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        metrics = self._calculate_metrics(all_labels, all_preds)

        return epoch_loss, epoch_acc, metrics

    def validate(self, val_loader: DataLoader, num_classes: int = 3, inc_metrics: bool=False) -> Tuple[float, float, Dict]:
        class_weights = self._compute_class_weights(val_loader, num_classes)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)
        val_metrics = self._calculate_validation_metrics(all_labels, all_preds, all_probs)
        
        if inc_metrics:
            self.print_metrics(all_labels, all_preds, all_probs)

        return val_loss, val_acc, val_metrics
    
    
    
    def predict(self, tensor_image: torch.Tensor) -> Tuple[int, float]:
        self.model.eval()
        with torch.no_grad():
            tensor_image = tensor_image.to(self.device)
            output = self.model(tensor_image)
            _, predicted_class = torch.max(output, 1)
            predicted_class = predicted_class.item()
            confidence = torch.softmax(output, dim=1)[0][predicted_class].item()
            return predicted_class, confidence

    
    def kfold_cross_validation(self, dataset: image_dataset.FundusDataset, num_epochs: int = 10, n_splits: int = 10, batch_size: int = 32):
        print(f"Number of images: {len(dataset.image_paths)}")
        
        image_transform = image_transforms.FundusImageTransforms()
        train_transform, test_transform = image_transform.train_transform, image_transform.test_transform

        labels = self._get_labels_for_kfold(dataset)
        
        # Group indices by class
        class_indices = {cls: np.where(np.array(labels) == cls)[0] for cls in set(labels)}
        
        # Create folds ensuring each class is represented
        folds = [[] for _ in range(n_splits)]
        for cls, indices in class_indices.items():
            np.random.shuffle(indices)
            fold_size = len(indices) // n_splits
            for i in range(n_splits):
                start = i * fold_size
                end = start + fold_size if i < n_splits - 1 else None
                folds[i].extend(indices[start:end])
        
        full_dataset = image_dataset.FundusDataset(
            image_dir=dataset.image_dir,
            csv_file=proj_util.TRAIN_LABEL_PATH,
            transform=train_transform,
            test=False,
            num_augmentations=dataset.num_augmentations
        )

        fold_results = []
        for fold, val_idx in enumerate(folds):
            train_idx = np.concatenate([folds[i] for i in range(n_splits) if i != fold])
            
            print('-' * 50)
            print(f'Fold {fold+1}/{n_splits}')
            
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)
            val_subset.dataset.transform = test_transform
            val_subset.dataset.train = False
            val_subset.dataset.num_augmentations = 1

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            best_val_loss = float('inf')
            best_model_state = None

            for epoch in range(num_epochs):
                train_loss, train_acc, _ = self.train_one_epoch(train_loader=train_loader)
                val_loss, val_acc, val_metrics = self.validate(val_loader=val_loader)

                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)

                print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()

                if val_acc > self.best_accuracy:
                    self.best_accuracy = val_acc
                    self.save_model()

                if self.early_stopper(val_loss):
                    print("Early stopping")
                    break

            fold_results.append((train_loss, val_loss, val_metrics))
        
        # # Calculate and print average metrics across folds
        # avg_train_loss = np.mean([result[0] for result in fold_results])
        # avg_val_loss = np.mean([result[1] for result in fold_results])
        # avg_val_metrics = {
        #     metric: np.mean([result[2][metric] for result in fold_results])
        #     for metric in fold_results[0][2] if metric != 'confusion_matrix'
        # }
        
        # print(f'\nAverage results across {n_splits} folds:')
        # print(f'Average Train Loss: {avg_train_loss:.4f}')
        # print(f'Average Validation Loss: {avg_val_loss:.4f}')
        # print('Average Validation Metrics:')
        # for metric, value in avg_val_metrics.items():
        #     print(f'{metric}: {value:.4f}')

    
    
    
    
    
    
    
    
    
    

    def _initialize_training(self) -> None:
        self.model = cnn_models.CNNModel(model_name=self.model_name).model
        self.model.to(self.device)
        self.optimizer = self._get_optimizer(lr=0.001, betas=(0.5, 0.99), weight_decay=1e-05)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    
    def _get_optimizer(self, lr: float, weight_decay: float, betas: Tuple[float, float]) -> optim.Optimizer:
        if self.model_name == "densenet121":
            return optim.Adam([
                {'params': self.model.model.features.parameters(), 'lr': lr, 'weight_decay': weight_decay},
                {'params': self.model.model.classifier.parameters(), 'lr': lr, 'weight_decay': weight_decay}
            ], betas=betas)
        elif self.model_name == "resnext50":
            return optim.Adam([
                {'params': self.model.model.fc.parameters(), 'lr': lr, 'weight_decay': weight_decay},
                {'params': [param for name, param in self.model.model.named_parameters() if "fc" not in name], 'lr': lr, 'weight_decay': weight_decay}
            ], betas=betas)
        else:
            raise ValueError("Invalid model name")

    def _compute_class_weights(self, dataloader: DataLoader, num_classes: int) -> torch.Tensor:
        all_labels = []
        for _, labels in dataloader:
            all_labels.extend(labels.cpu().numpy())

        all_labels = np.array(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)
        print("Class distribution:", dict(zip(unique, counts)))
        # Initialize weights for all classes
        class_weights = np.ones(num_classes)
        
        # Compute weights only for classes present in the data
        present_class_weights = compute_class_weight('balanced', classes=unique, y=all_labels)
        
        # Assign computed weights to present classes
        for i, cls in enumerate(unique):
            class_weights[cls] = present_class_weights[i]

        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        print("Computed class weights:", class_weights)

        return class_weights


    def _calculate_metrics(self, labels: List[int], preds: List[int]) -> Dict[str, float]:
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)

        cm = confusion_matrix(labels, preds)
        sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        return {'precision': precision, 'recall': recall, 'f1': f1, 'sensitivity': sensitivity}

    def _calculate_validation_metrics(self, labels: List[int], preds: List[int], probs: List[List[float]]) -> Dict[str, float]:
        cm = confusion_matrix(labels, preds)
        # auc = roc_auc_score(labels, probs, multi_class='ovo')
        # auc = 0
        probs = np.array(probs)
        labels = np.array(labels)
    
        n_classes = probs.shape[1]
        unique_labels = np.unique(labels)
        
        if len(unique_labels) != n_classes:
            print(f"Warning: Number of unique labels ({len(unique_labels)}) doesn't match number of classes in probabilities ({n_classes})")
            auc = None
        elif n_classes == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            labels_onehot = np.eye(n_classes)[labels]
            auc = roc_auc_score(labels_onehot, probs, multi_class='ovr', average='macro')
        
        precision = precision_score(labels, preds, average='weighted', zero_division=1)
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')
        sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0

        return {
            'confusion_matrix': cm,
            'roc_auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sensitivity': sensitivity
        }

    def save_model(self) -> None:
        scripted_model = torch.jit.script(self.model)
        torch.jit.save(scripted_model, self.model_path)
        print(f'Model saved to {self.model_path}')

    def load_model(self):
        if os.path.exists(self.model_path):
            loaded_model = torch.jit.load(self.model_path)
            self.model.load_state_dict(loaded_model.state_dict())
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"No existing model found at {self.model_path}. Starting with a fresh model.")

    
    

    def plot_learning_rates(self) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.learning_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate over Epochs')
        plt.yscale('log')
        plt.grid(True)
        plt.show()

    def _get_labels_for_kfold(self, dataset: image_dataset.FundusDataset) -> List[int]:
        return [dataset._get_labels(path).item() for path in dataset.image_paths]

    def _print_epoch_results(self, epoch: int, num_epochs: int, train_loss: float, train_acc: float,
                             val_loss: float, val_acc: float, val_metrics: Dict, current_lr: float, epoch_duration: float) -> None:
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
        print(f'ROC AUC: {val_metrics["roc_auc"]:.4f}')
        print(f'Current Learning Rate: {current_lr:.6f}')
        print(f'Time taken for epoch: {epoch_duration:.2f} seconds')
        
    def print_formatted_metrics(self, val_metrics):
        print("Validation Metrics:")
        print("-------------------")
        for metric, value in val_metrics.items():
            if metric == 'confusion_matrix':
                print(f"{metric}:\n{value}")
            else:
                print(f"{metric}: {value:.4f}")
                
                
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

        probabilities = np.array(probabilities)
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

        
        labels_onehot = np.eye(n_classes)[labels]
        overall_roc_auc = roc_auc_score(labels_onehot, probabilities, average='macro', multi_class='ovr')
        print(f"Overall ROC AUC: {overall_roc_auc:.4f}")