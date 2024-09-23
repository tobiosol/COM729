import os
import cv2
import numpy as np
import torch
print(torch.version.cuda)
import cupy as cp
print(f"CuPy Version: {cp.__version__}")
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy as entropy
import matplotlib.pyplot as plt

class FundusImageProcessor:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def calculate_metrics(self, original_image, enhanced_image):
        
        enhanced_image = cv2.resize(enhanced_image, (original_image.shape[1], original_image.shape[0]))
        
        # Convert enhanced_image to NumPy array for CPU operations
        enhanced_image_np = np.asarray(enhanced_image)

        # Calculate the necessary padding
        pad_height = max(0, original_image.shape[0] - enhanced_image_np.shape[0])
        pad_width = max(0, original_image.shape[1] - enhanced_image_np.shape[1])

        # Pad enhanced image if necessary
        enhanced_image_np = np.pad(enhanced_image_np, ((pad_height // 2, pad_height // 2),
                                                       (pad_width // 2, pad_width // 2)),mode='constant')
        
        
        # Ensure both images have the same data type (e.g., float32)
        original_image = original_image.astype(np.float32)
        enhanced_image_np = enhanced_image_np.astype(np.float32)
        
        cii = np.std(enhanced_image_np) / np.std(original_image)
        
        ssim_value = ssim(original_image, enhanced_image_np)
        hist, _ = np.histogram(enhanced_image_np, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        entropy_value = entropy(hist)
        
        
        
        # Calculate PSNR
        mse = np.mean((original_image - enhanced_image_np) ** 2)
        psnr_value = 20 * np.log10(255.0 / np.sqrt(mse))
        
        mean_std_ratio_value = np.mean(enhanced_image_np) / np.std(enhanced_image_np)

        # psnr_value = cv2.PSNR(original_image, enhanced_image_np)

        print(f"SSIM: {ssim_value:.4f}, Entropy: {entropy_value:.4f}, PSNR: {psnr_value:.2f}, Mean/Std: {mean_std_ratio_value:.4f}, CII: {cii:.4f}")
        return cii, ssim_value, entropy_value, mean_std_ratio_value

    def adaptive_histogram_equalization(self, image):
        return cv2.equalizeHist(image)

    def gaussian_blur(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def preprocess(self, image, target_radius=300):
        # original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        original_image = image
        scaled_image = self.scale_radius(original_image, target_radius)
        local_mean = cv2.GaussianBlur(scaled_image, (0, 0), sigmaX=10)
        enhanced_image = scaled_image - local_mean + 128

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
        clahe_enhanced_image = clahe.apply(enhanced_image)

        clipped_image = clahe_enhanced_image * (clahe_enhanced_image > 128)
        normalized_image = (clipped_image - np.min(clipped_image)) / (np.max(clipped_image) - np.min(clipped_image))
        
        image_title_pairs = [
            (original_image, 'Original'),
            (scaled_image, 'scaled_image'),
            (local_mean, 'local_mean'),
            (enhanced_image, 'enhanced_image'),
            (clahe_enhanced_image, 'clahe_enhanced_image'),
            (clipped_image, 'clipped_image'),
            (normalized_image, 'normalized_image')
        ]
        
        # self.plot_images(image_title_pairs)

        return clahe_enhanced_image

    def scale_radius(self, img, scale):
        # Compute the sum of pixel intensities along the vertical axis
        x = img.sum(axis=1)
        # Estimate the retinal radius based on the sum
        r = (x > x.mean() / 10).sum() // 2
        # Calculate the scaling factor
        s = scale * 1.0 / r
        # Resize the image
        return cv2.resize(img, (0, 0), fx=s, fy=s)

    def preprocess_and_save_dataset(self, src_image_dir, dest_image_dir):
        image_paths = self.load_images(src_image_dir)
        for image_path in image_paths:
            original_filename = os.path.basename(image_path)
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            processed_image = self.preprocess_image(original_image)
            if processed_image is not None:
                self.save_processed_image(processed_image, dest_image_dir, original_filename)
            else:
                print(f"Skipping {original_filename} due to preprocessing error.")

    def load_images(self, image_folder):
        return [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.lower().endswith((".jpg", ".png"))]

    def save_processed_image(self, image, dest_image_dir, original_filename):
        new_filename = os.path.splitext(original_filename)[0] + os.path.splitext(original_filename)[1]
        save_path = os.path.join(dest_image_dir, new_filename)
        image_np = cp.asnumpy(image)
        cv2.imwrite(save_path, image_np)
        
    def plot_images(self, image_title_pairs):
        """
        Plots a list of images along with their titles.

        Args:
            image_title_pairs (list of tuples): Each tuple contains an image (numpy array) and its title.
        """
        num_images = len(image_title_pairs)
        rows = int(num_images ** 0.5)  # Square grid layout
        cols = (num_images + rows - 1) // rows

        plt.figure(figsize=(12, 6))
        for i, (image, title) in enumerate(image_title_pairs, start=1):
            plt.subplot(rows, cols, i)
            plt.imshow(image, cmap='gray')
            plt.title(title)

        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":
    preprocessor = FundusImageProcessor()
    image_path = 'timg/IMG0413 (8).png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    processed_image = preprocessor.preprocess(image)

    # Example usage:
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    metrics = preprocessor.calculate_metrics(original_image, processed_image)
    print(metrics)

from numpy.typing import NDArray
