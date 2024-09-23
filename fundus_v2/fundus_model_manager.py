import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))
subdirectory_path = Path(current_dir) / 'fundus_v2'
sys.path.append(str(subdirectory_path))

import cv2
import torch
from typing import List
from PIL import Image

from fundus_v2 import (
    cnn_models,
    cnn_trainer,
    image_loader,
    image_transforms,
    random_forest_classifier,
    svm_classifier,
    ensemble_classifier,
    image_preprocessor,
    proj_util
)

class FundusModelManager:
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.augmentor = image_transforms.FundusImageTransforms()
        self.train_transform = self.augmentor.train_transform
        self.test_transform = self.augmentor.test_transform
        self.class_labels = ["Cotton-wool Spots", "Diabetic Retinopathy", "No Identifiable Eye Disease"]
        self.model_names: List[str] = ['densenet121', 'resnext50']
        
        self.train_loader = self._create_data_loader(
            image_dir=proj_util.TRAINING_DIR,
            csv_file=proj_util.TRAIN_LABEL_PATH,
            transform=self.train_transform,
            train=True,
            num_augmentations=5
        )
        
        self.val_loader = self._create_data_loader(
            image_dir=proj_util.VALIDATION_DIR,
            csv_file=proj_util.VALIDATION_LABEL_PATH,
            transform=self.test_transform,
            train=False,
            num_augmentations=1
        )

        self.cnn_models = [cnn_models.CNNModel(name) for name in self.model_names]
    
    def _create_data_loader(self, image_dir, csv_file, transform, train, num_augmentations):
        dataloader = image_loader.FundusImageLoader(
            image_dir=image_dir,
            csv_file=csv_file,
            batch_size=32,
            transform=transform,
            shuffle=train,
            train=train,
            num_augmentations=num_augmentations
        )
        return dataloader.get_loader()
    
    def train_cnn_models(self):
        """Train CNN models"""
        for model_name in self.model_names:
            trainer = cnn_trainer.FundusCNNTrainer(
                model_name=model_name,
                train_loader=self.train_loader,
                val_loader=self.val_loader
            )
            trainer.train(num_epochs=10)
            val_loss, val_acc, val_metrics = trainer.validate(self.val_loader)
            print(f'Model: {model_name}')
            print(f'Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc:.4f}')
            print(f'Validation Metrics: {trainer.print_formatted_metrics(val_metrics)}')
            print('-' * 50)
            
    def train_cnn_models_kfold(self):        
        """Train CNN models with k-fold cross-validation"""
        for model_name in self.model_names:
            trainer = cnn_trainer.FundusCNNTrainer(
                model_name=model_name,
                train_loader=self.train_loader,
                val_loader=self.val_loader
            )
            
            print(f"Training {model_name} with k-fold cross-validation...")
            trainer.kfold_cross_validation(dataset=self.train_loader.dataset, n_splits=5)            
            val_loss, val_acc, val_metrics = trainer.validate(self.val_loader, inc_metrics=True)
            print(f'Model: {model_name}')
            print(f'Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc:.4f}')
            print(f'Validation Metrics: {trainer.print_formatted_metrics(val_metrics)}')
            print('-' * 50)
            
    def train_random_forest(self):
        """Train Random Forest model"""
        rf_classifier = random_forest_classifier.RandomForestFeatureClassifier(
            cnn_models=self.cnn_models,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device
        )
        
        # rf_classifier.train()
        # rf_classifier.evaluate()
        
        
        preprocessor = image_preprocessor.FundusImageProcessor()
        # image_path = 'timg/IMG0413 (8).png'
        image_path = 'timg/116.png'
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        processed_image = preprocessor.preprocess(image)  
        transformed_image = self.test_transform(Image.fromarray(processed_image))
        
        # Convert image to tensor
        image_tensor = torch.tensor(np.array(transformed_image), dtype=torch.float32).squeeze()
        image_tensor = image_tensor.repeat(1, 1, 1)
        image_tensor = image_tensor.unsqueeze(0)
        print(rf_classifier.predict(image_tensor.to(self.device)))
    
    def train_svm(self):
        """Train SVM model"""
        s_classifier = svm_classifier.SVMFeatureClassifier(
            cnn_models=self.cnn_models,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device
        )
        
        s_classifier.train()
        s_classifier.evaluate()
        
        
        preprocessor = image_preprocessor.FundusImageProcessor()
        # image_path = 'timg/IMG0413 (8).png'
        image_path = 'timg/116.png'
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        processed_image = preprocessor.preprocess(image)        
        transformed_image = self.test_transform(Image.fromarray(processed_image))
        
        # Convert image to tensor
        image_tensor = torch.tensor(np.array(transformed_image), dtype=torch.float32).squeeze()
        image_tensor = image_tensor.repeat(1, 1, 1)
        image_tensor = image_tensor.unsqueeze(0)
        print(s_classifier.predict(image_tensor.to(self.device)))
    
    
    def predict(self, file, selected_model):
        """Predict the class of an image"""
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        preprocessor = image_preprocessor.FundusImageProcessor()
        processed_image = preprocessor.preprocess(image)
        transformed_image = self.test_transform(Image.fromarray(processed_image))
        
        # Convert image to tensor
        image_tensor = torch.tensor(np.array(transformed_image), dtype=torch.float32).squeeze()
        image_tensor = image_tensor.repeat(1, 1, 1)
        image_tensor = image_tensor.unsqueeze(0)
        
        s_classifier = svm_classifier.SVMFeatureClassifier(
            cnn_models=self.cnn_models,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device
        )
        
        predicted_class, prediction_matrix = s_classifier.predict(image_tensor.to(self.device))
        print(f"Predicted class: {predicted_class}")
        print(f"Prediction matrix: {prediction_matrix}")
        predicted_label = self.class_labels[predicted_class]
        
        return predicted_label, prediction_matrix
    
    def train_ml_ensemble(self):
        """Train Ensemble model"""
        en_classifier = ensemble_classifier.EnsembleClassifier(
            cnn_models=self.cnn_models,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device
        )
        
        en_classifier.train()
        en_classifier.evaluate()
        
        
        preprocessor = image_preprocessor.FundusImageProcessor()
        # image_path = 'timg/IMG0413 (8).png'
        image_path = 'timg/116.png'
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        processed_image = preprocessor.preprocess(image)        
        transformed_image = self.test_transform(Image.fromarray(processed_image))
        
        # Convert image to tensor
        image_tensor = torch.tensor(np.array(transformed_image), dtype=torch.float32).squeeze()
        
        print(en_classifier.predict(image_tensor.to(self.device)))

if __name__ == "__main__":
    manager = FundusModelManager()
    manager.train_cnn_models_kfold()
    # manager.train_random_forest()
    # manager.train_svm()
    # manager.train_ml_ensemble()
