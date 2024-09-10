import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np
from choose_device import choose_device


# Load the Real-ESRGAN model
def load_model():
    device = choose_device()

    # RealESRGAN model parameters
    realesrgan_model_path = 'pretrained_models/RealESRGAN_x4plus.pth'
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    # Initialize RealESRGANer with the model
    upscaler = RealESRGANer(scale=4, model_path=realesrgan_model_path, model=model, tile=0, tile_pad=10, pre_pad=0,
                            half=False, device=device)

    return upscaler


def upscale_image(input_image: np.ndarray) -> np.ndarray:
    # Convert image to RGB (if needed)
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Load model
    upscaler = load_model()

    # Upscale the image
    upscaled_img, _ = upscaler.enhance(input_image_rgb, outscale=4)

    upscaled_img_bgr = cv2.cvtColor(upscaled_img, cv2.COLOR_RGB2BGR)

    return upscaled_img_bgr


# Example usage
if __name__ == "__main__":
    # Read the input image
    input_image = cv2.imread('monarch_LR4.png')

    if input_image is None:
        raise ValueError("Failed to load the input image. Check the file path and format.")

    # Process the image
    output_image = upscale_image(input_image)

    # Save the output image
    cv2.imwrite('real_esrgan_upscale.jpg', output_image)
