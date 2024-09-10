import cv2
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
import numpy as np

from gfpganModel import GFPGANer
from realesrgan import RealESRGANer
from choose_device import choose_device


# Load the GFPGAN model
def load_models():
    device = choose_device()
    # Load RealESRGAN model
    realesrgan_model_path = 'pretrained_models/RealESRGAN_x4plus.pth'
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upscaler = RealESRGANer(scale=4, model_path=realesrgan_model_path, model=model, tile=0, tile_pad=10, pre_pad=0,
                            half=False, device=device)

    # Load GFPGAN model
    model_path = 'pretrained_models/GFPGANv1.3.pth'
    restorer = GFPGANer(model_path=model_path, upscale=4, arch='clean', channel_multiplier=2, bg_upsampler=upscaler,
                        device=device)

    return restorer


def gfpgan_upscale(input_image: np.ndarray) -> np.ndarray:
    # Load and preprocess the image
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Load models
    restorer = load_models()

    # Enhance the face image
    cropped_faces, restored_faces, restored_img = restorer.enhance(input_image_rgb, has_aligned=False,
                                                                   only_center_face=False, paste_back=True)

    # Convert back to BGR
    restored_img_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

    return restored_img_bgr


# Example usage
if __name__ == "__main__":
    # Read the input image
    input_image = cv2.imread('face_38.jpg')

    # Process the image
    output_image = gfpgan_upscale(input_image)

    # Save the output image
    cv2.imwrite('path_to_save_restored_face_image.jpg', output_image)
