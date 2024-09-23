import torch
import torch.nn as nn
from torchvision import models

class DenseNet121(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super().__init__()
        # self.model = models.densenet121(pretrained=True)
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.features.norm0 = nn.BatchNorm2d(64)
        num_ftrs = self.model.classifier.in_features
        
        feature_extractor = self.model.features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
        
        for param in self.model.parameters():
            param.requires_grad = True
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

class ResNeXt50(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super().__init__()
        # self.model = models.resnext50_32x4d(pretrained=True)
        self.model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        # Modify the first convolutional layer to accept 1 channel input
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

        # Unfreeze more layers
        for param in self.model.parameters():
            param.requires_grad = True

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

class CNNModel(nn.Module):
    def __init__(self, model_name, num_classes=3):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = self._initialize_model()

    def _initialize_model(self):
        print(f"Initializing {self.model_name} model...")
        if self.model_name == "densenet121":
            return DenseNet121(num_classes=self.num_classes)
        elif self.model_name == "resnext50":
            return ResNeXt50(num_classes=self.num_classes)
        else:
            raise ValueError("Invalid model name")

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def extract_features(self, image_tensor):
        features = self.model(image_tensor)
        return features.view(features.size(0), -1)
