import torch
import cv2
import numpy as np



def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def uint2tensor4(img):
    img = img.numpy()
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)

def tensor2uint(img):
    # Ensure img is a 4D tensor with shape [N, C, H, W] or a 3D tensor [C, H, W]
    if img.ndim == 4:
        img = img.squeeze(0)  # Remove batch dimension

    # Ensure tensor values are clamped to [0, 1] and convert to float
    img = img.float().clamp(0, 1).cpu().numpy()

    # Transpose to [H, W, C] if img has 3 dimensions
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    # Scale to [0, 255] and convert to uint8
    img = (img * 255.0).clip(0, 255)  # Clip to avoid values out of range due to rounding errors
    return np.uint8(img.round())

def tensor_to_uint8(tensor):
    """
    Convert a tensor to uint8 format for image representation.

    Args:
        tensor (torch.Tensor): The input tensor with values in range [0, 1].

    Returns:
        np.ndarray: Converted image array in uint8 format.
    """
    # Clamp tensor values to the range [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Scale to [0, 255]
    tensor = tensor * 255

    # Convert to numpy array and change dtype to uint8
    image_np = tensor.byte().cpu().numpy()

    return image_np