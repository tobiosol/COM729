
# NICL Detection System

## Project Structure

### fundus_v2/
Main directory containing the core components of the fundus image analysis project.

#### cnn_models.py
- Defines CNN model architectures for fundus image classification
- Supports DenseNet121 and ResNeXt50 models
- Implements a CNNModel class for easy model initialization and management

#### cnn_trainer.py
- Implements the FundusCNNTrainer class for training and evaluating CNN models
- Handles model initialization, training loops, and validation
- Supports k-fold cross-validation with class balance assurance
- Includes methods for metric calculation and visualization (confusion matrix, ROC AUC)

#### image_dataset.py
- Contains the FundusDataset class for handling fundus image datasets
- Manages image loading, label assignment, and data augmentation
- Supports both training and testing dataset configurations

#### image_transforms.py
- Defines image transformation operations for data augmentation and preprocessing
- Implements FundusImageTransforms class with separate train and test transforms
- Includes operations like resizing, normalization, and data augmentation techniques

#### proj_util.py
- Utility functions for file handling, path management, and other helper operations
- Includes functions for model file management, image loading, and filename extraction

#### random_forest_classifier.py
- Implements the RandomForestFeatureClassifier class for fundus image classification
- Utilizes scikit-learn's RandomForestClassifier
- Includes methods for feature extraction, model training, and evaluation
- Supports saving and loading of trained models

#### svm_classifier.py
- Implements the SVMFeatureClassifier class for fundus image classification
- Uses scikit-learn's SVM classifier with grid search for hyperparameter tuning
- Provides functionality for feature extraction, model training, and evaluation
- Includes methods for saving and loading trained models
## Key Features

- Multi-architecture support for fundus image classification
- Robust data handling and augmentation pipeline
- K-fold cross-validation with class balance optimization
- Comprehensive metric calculation and visualization
- Flexible utility functions for project management

## Usage

The FundusCNNTrainer class in cnn_trainer.py serves as the main entry point for model training and evaluation. Use this class to train models, perform cross-validation, and assess model performance on fundus images.
