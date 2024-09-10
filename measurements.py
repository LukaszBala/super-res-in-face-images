import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from pyiqa import create_metric
import torchvision.transforms as transforms
from PIL import Image


# Function to calculate PSNR
def calculate_psnr(img1, img2):
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    psnr_value = compare_psnr(img1, img2)
    return psnr_value


# Function to calculate SSIM
def calculate_ssim(img1, img2):
    """Calculate the Structural Similarity Index (SSIM) between two images."""
    ssim_value = compare_ssim(img1, img2, multichannel=True, channel_axis=2)
    return ssim_value


# Function to calculate NIQE using pyiqa
def calculate_niqe(image):
    """Calculate the NIQE score of an image."""
    niqe_metric = create_metric('niqe')
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    niqe_value = niqe_metric(image_tensor).item()
    return niqe_value


# Function to calculate BRISQUE using pyiqa
def calculate_brisque(image):
    """Calculate the BRISQUE score of an image."""
    brisque_metric = create_metric('brisque')
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    brisque_value = brisque_metric(image_tensor).item()
    return brisque_value


# Function to load and process an image
def load_and_process_image(image_path):
    """Load an image and convert it to RGB if needed."""
    image = Image.open(image_path).convert('RGB')
    return image


# Function to evaluate image quality metrics based on image paths
def evaluate_image_quality_from_paths(original_image_path, upscaled_image_path):
    """Evaluate PSNR, SSIM, NIQE, and BRISQUE for images loaded from file paths."""
    # Load the images
    original_image = load_and_process_image(original_image_path)
    upscaled_image = load_and_process_image(upscaled_image_path)

    # Convert original_image to numpy array for PSNR and SSIM calculations
    original_image_np = np.array(original_image)
    upscaled_image_np = np.array(upscaled_image)

    # Calculate PSNR and SSIM
    psnr_value = calculate_psnr(original_image_np, upscaled_image_np)
    ssim_value = calculate_ssim(original_image_np, upscaled_image_np)

    # Calculate NIQE and BRISQUE scores
    niqe_value = calculate_niqe(upscaled_image)
    brisque_value = calculate_brisque(upscaled_image)

    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'NIQE': niqe_value,
        'BRISQUE': brisque_value
    }


# Function to evaluate image quality metrics based on already loaded images
def evaluate_image_quality_from_images(original_image, upscaled_image):
    """Evaluate PSNR, SSIM, NIQE, and BRISQUE for already loaded images."""
    # Convert original_image to numpy array for PSNR and SSIM calculations
    original_image_np = np.array(original_image)
    upscaled_image_np = np.array(upscaled_image)

    # Calculate PSNR and SSIM
    psnr_value = calculate_psnr(original_image_np, upscaled_image_np)
    ssim_value = calculate_ssim(original_image_np, upscaled_image_np)

    # Calculate NIQE and BRISQUE scores
    niqe_value = calculate_niqe(upscaled_image)
    brisque_value = calculate_brisque(upscaled_image)

    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'NIQE': niqe_value,
        'BRISQUE': brisque_value
    }


if __name__ == "__main__":
    # Example Usage with file paths
    original_image_path = 'monarch_HR.png'  # Path to the original image
    upscaled_image_path = 'real_esrgan_upscale.jpg'  # Path to the upscaled image

    # Evaluate image quality metrics
    results_from_paths = evaluate_image_quality_from_paths(original_image_path, upscaled_image_path)
    print("Results from image paths:")
    print(f"PSNR: {results_from_paths['PSNR']:.2f} dB")
    print(f"SSIM: {results_from_paths['SSIM']:.4f}")
    print(f"NIQE: {results_from_paths['NIQE']:.4f}")
    print(f"BRISQUE: {results_from_paths['BRISQUE']:.4f}")

    # Example Usage with already loaded images
    original_image = load_and_process_image(original_image_path)
    upscaled_image = load_and_process_image(upscaled_image_path)

    # Evaluate image quality metrics
    results_from_images = evaluate_image_quality_from_images(original_image, upscaled_image)
    print("\nResults from already loaded images:")
    print(f"PSNR: {results_from_images['PSNR']:.2f} dB")
    print(f"SSIM: {results_from_images['SSIM']:.4f}")
    print(f"NIQE: {results_from_images['NIQE']:.4f}")
    print(f"BRISQUE: {results_from_images['BRISQUE']:.4f}")
