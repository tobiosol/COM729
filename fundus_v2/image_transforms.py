import torchvision.transforms as transforms

class FundusImageTransforms:
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-20, 20)),
            transforms.ColorJitter(brightness=0.05, contrast=0.02, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=0.1),
            transforms.ToTensor(),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.test_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
