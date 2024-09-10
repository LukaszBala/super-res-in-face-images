import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .rlfn import RLFN_S
from utils import tensor2uint
from choose_device import choose_device

# Set device and model parameters

# Load the model
def load_model(device, model_path):
    """Load the pre-trained model."""
    model = RLFN_S(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Preprocess the input PIL image
def preprocess_image(image, device):
    """
    Preprocess the input PIL image.

    Args:
        image (PIL.Image): Input PIL image.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device)

# Perform super-resolution on the provided image
def super_resolve_image(image):
    """
    Super-resolve a PIL image using the pre-trained model.

    Args:
        image (PIL.Image): Input low-resolution image.

    Returns:
        PIL.Image: Super-resolved image.
    """
    # Load the model

    device = choose_device()
    model_path = "epoch_2500.pth"  # Path to your trained model
    upscale_factor = 4  # Upscaling factor

    model = load_model(device, model_path)

    # Preprocess the input image
    input_image = preprocess_image(image, device)

    # Run the model on the preprocessed image
    with torch.no_grad():
        SR_image = model(input_image)

    # Postprocess the output image
    output_image = tensor2uint(SR_image.squeeze(0).cpu())  # Remove batch dimension and move to CPU
    output_image = Image.fromarray(output_image)
    output_array = np.array(output_image)

    return output_array

# Example usage
if __name__ == "__main__":
    image_path = 'monarch.png'  # Replace with the path to your low-res input image
    image = Image.open(image_path).convert('RGB')
    upscale_factor = 4

    # Perform super-resolution
    output_image = super_resolve_image(image)

    # Convert input PIL image to numpy array for comparison
    original_image = np.array(image)
    original_height, original_width = original_image.shape[:2]

    new_width = int(original_width * upscale_factor)
    new_height = int(original_height * upscale_factor)

    # Bicubic interpolation for comparison
    bicubic_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Plot the input, output, and bicubic images
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original_image)
    axes[0].set_title('Low-Resolution Input')
    axes[0].axis('off')

    axes[1].imshow(output_image)
    axes[1].set_title('Super-Resolved Output')
    axes[1].axis('off')

    axes[2].imshow(bicubic_image)
    axes[2].set_title('Bicubic Interpolation')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    output_image.save('super_resolved_image.png')  # Save the super-resolved image
