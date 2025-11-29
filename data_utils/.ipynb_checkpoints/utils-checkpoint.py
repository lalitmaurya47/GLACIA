import pandas as pd
import numpy as np
import os
from random import shuffle
import random
from PIL import Image
import requests
from io import BytesIO
from sklearn.utils import shuffle
#import tifffile  # handles multi-band TIFFs better than PIL
import rasterio
from rasterio.io import MemoryFile
import torch

# def normalize_image(image):
#     """Normalize a single image to [0, 1]."""
#     img_std = np.std(image)
#     img_mean = np.mean(image)
#     img_normalized = (image - img_mean) / img_std
#     img_normalized = ((img_normalized - np.min(img_normalized)) / 
#                       (np.max(img_normalized) - np.min(img_normalized))) * 255
#     return img_normalized

# def load_annotation(annotation_path):
#     list_df_path = os.listdir(annotation_path)
#     list_df_path = [os.path.join(annotation_path, df) for df in list_df_path]
#     list_df = []
#     for df_path in list_df_path:
#         df = pd.read_csv(df_path)
#         df = df.dropna()
#         df = df.reset_index(drop=True)
#         list_df.append(df)
#     df = pd.concat(list_df, ignore_index=True)
#     df = df.dropna()
#     # df = df.sample(frac=1, random_state=42)
#     df = shuffle(df)
#     df = df.reset_index(drop=True)
#     train_df = df[df['split'] == 'train']
#     test_df = df[df['split'] == 'test']
#     return train_df, test_df
def load_annotation(annotation_path):
    list_df = []
    for df_name in os.listdir(annotation_path):
        df_path = os.path.join(annotation_path, df_name)

        # ✅ Skip directories like `.ipynb_checkpoints`
        if os.path.isdir(df_path):
            continue

        # ✅ Optionally ensure it's a CSV file
        if not df_path.endswith(".csv"):
            continue

        df = pd.read_csv(df_path)
        df = df.dropna().reset_index(drop=True)
        list_df.append(df)

    df = pd.concat(list_df, ignore_index=True)
    df = shuffle(df.dropna()).reset_index(drop=True)
    
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    return train_df, test_df
#def load_image(image_file):
#    if image_file.startswith('http://') or image_file.startswith('https://'):
#        response = requests.get(image_file)
#        image = Image.open(BytesIO(response.content)).convert('RGB')
#    else:
#        image = Image.open(image_file).convert('RGB')
#    return image

# def load_image(image_file, channel_indices=[0, 1, 2, 3, 4, 5]):
#     """
#     Loads a multi-channel image (local or URL) and selects specific channels.

#     Args:
#         image_file (str): Path or URL of the image file (.tif or others)
#         channel_indices (list[int]): Which channel indices to keep (0-based)
#     Returns:
#         np.ndarray: Image array of shape (H, W, len(channel_indices))
#     """
#     # Load image data (supports both URL and local)
#     if image_file.startswith(('http://', 'https://')):
#         response = requests.get(image_file)
#         image_data = BytesIO(response.content)
#     else:
#         image_data = image_file

#     # Read using tifffile for multi-channel TIFFs
#     image = tifffile.imread(image_data)

#     # Handle grayscale or lower-channel images safely
#     if image.ndim == 2:
#         image = np.expand_dims(image, axis=-1)

#     # Select the requested channel indices (ensure valid range)
#     channel_indices = [i for i in channel_indices if i < image.shape[-1]]
#     image = image[..., channel_indices]

#     # Optionally convert to float32 and normalize
#     image = image.astype(np.float32)
#     image -= image.min()
#     image /= (image.max() + 1e-8)
#     #print("image", image.shape)
#     tifffile.imwrite("output_image.tif", image)

#     return image

# def load_image(image_file, channel_indices=[0, 1, 2, 3, 4, 5]):
#     """
#     Loads a multi-channel image (local or URL) and selects specific channels using rasterio.

#     Args:
#         image_file (str): Path or URL of the image file (.tif or others)
#         channel_indices (list[int]): Which channel indices to keep (0-based)
#     Returns:
#         np.ndarray: Image array of shape (H, W, len(channel_indices))
#     """

#     # Load remote or local image
#     if image_file.startswith(('http://', 'https://')):
#         response = requests.get(image_file)
#         response.raise_for_status()
#         memfile = MemoryFile(response.content)
#         src = memfile.open()
#     else:
#         src = rasterio.open(image_file)

#     # Validate and adjust channel indices (1-based in rasterio)
#     total_bands = src.count
#     valid_indices = [i + 1 for i in channel_indices if i < total_bands]

#     if not valid_indices:
#         raise ValueError(f"No valid channels found in {image_file}. Available: {total_bands}")

#     # Read selected bands
#     image = src.read(valid_indices)  # Shape: (C, H, W)
#     src.close()
#     # Normalize image and mask
#     image = normalize_image(image)

#     # Reorder to (H, W, C)
#     image = np.transpose(image, (1, 2, 0))

#     # # Handle normalization safely
#     # image = image.astype(np.float32)
#     # image -= image.min()
#     # image /= (image.max() + 1e-8)

#     # Save normalized image (optional)
#     #rasterio.imwrite("output_image.tif", np.transpose(image, (2, 0, 1)))

#     return image

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Safely normalize an image to [0, 1] range per channel.

    Args:
        image (np.ndarray): Image array of shape (C, H, W).
    Returns:
        np.ndarray: Normalized image of shape (C, H, W).
    """
    image = image.astype(np.float32)
    for c in range(image.shape[0]):
        channel = image[c]
        min_val, max_val = channel.min(), channel.max()
        if np.isfinite(min_val) and np.isfinite(max_val) and max_val > min_val:
            image[c] = (channel - min_val) / (max_val - min_val)
        else:
            image[c] = 0.0
    return image


def load_image(image_file: str, channel_indices=None) -> np.ndarray:
    """
    Loads a multi-channel image (local or URL) and selects specific channels using rasterio.

    Args:
        image_file (str): Path or URL of the image file (.tif or similar)
        channel_indices (list[int], optional): 0-based indices of channels to keep.
            Defaults to all available channels.
    Returns:
        np.ndarray: Normalized image array of shape (H, W, C).
    """
    # Load remote or local image
    if image_file.startswith(('http://', 'https://')):
        response = requests.get(image_file)
        response.raise_for_status()
        with MemoryFile(response.content) as memfile:
            with memfile.open() as src:
                total_bands = src.count
                if channel_indices is None:
                    channel_indices = list(range(total_bands))
                valid_indices = [i + 1 for i in channel_indices if 0 <= i < total_bands]
                if not valid_indices:
                    raise ValueError(f"No valid channels found in {image_file}. Available: {total_bands}")
                image = src.read(valid_indices)
    else:
        with rasterio.open(image_file) as src:
            total_bands = src.count
            if channel_indices is None:
                channel_indices = list(range(total_bands))
            valid_indices = [i + 1 for i in channel_indices if 0 <= i < total_bands]
            if not valid_indices:
                raise ValueError(f"No valid channels found in {image_file}. Available: {total_bands}")
            image = src.read(valid_indices)

    # Normalize and transpose to (H, W, C)
    image = normalize_image(image)
    image = np.transpose(image, (1, 2, 0))
    return image



# def load_image_rgb(image_path, channel_indices=[2,1,0]):
#     # Read TIFF
#     img = tifffile.imread(image_path)  # (H, W, C)

#     # Pick 3 channels
#     img = img[..., channel_indices]

#     # Convert to uint8 if needed
#     if img.dtype != np.uint8:
#         img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

#     # Convert to PIL Image
#     image_pil = Image.fromarray(img)
#     #print("image_pil size:", image_pil.size) 
#     image_pil.save("output_image.png")
#     return image_pil

def load_image_rgb(image_path, channel_indices=[2, 1, 0]):
    """
    Loads a multi-channel image using rasterio, selects specific RGB channels,
    converts to uint8, and returns a PIL Image.

    Args:
        image_path (str): Path or URL of the image file (.tif)
        channel_indices (list[int]): 0-based indices of bands to select for RGB

    Returns:
        PIL.Image: Image object with selected channels
    """
    # Load remote or local image
    if image_path.startswith(('http://', 'https://')):
        response = requests.get(image_path)
        response.raise_for_status()
        memfile = MemoryFile(response.content)
        src = memfile.open()
    else:
        src = rasterio.open(image_path)

    # Validate channels (rasterio uses 1-based indexing)
    total_bands = src.count
    valid_indices = [i + 1 for i in channel_indices if i < total_bands]
    if not valid_indices:
        raise ValueError(f"No valid channels found in {image_path}. Available: {total_bands}")

    # Read selected bands (C, H, W)
    img = src.read(valid_indices)
    src.close()

    # Convert to H, W, C
    img = np.transpose(img, (1, 2, 0))

    # Normalize and convert to uint8
    #img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
     # Normalize for display (robust percentile stretch)
    rgb = img
    rgb_min, rgb_max = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - rgb_min) / (rgb_max - rgb_min) * 255, 0, 255).astype(np.uint8)
    img = rgb
    
    # Convert to PIL Image
    image_pil = Image.fromarray(img)
    #image_pil.save("output_image.png")

    return image_pil

def binary_loader(mask_path):
    with open(mask_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

# def binary_loader(mask_path):
#     with open(mask_path, 'rb') as f:
#         img = Image.open(f).convert('L')  # convert to grayscale
#         mask = np.array(img, dtype=np.float32) / 255.0  # normalize to [0, 1]
#         return mask
# def binary_loader(mask_path):
#     """
#     Load a binary mask as a PIL image and normalize pixel values to [0, 1].
#     """
#     with open(mask_path, 'rb') as f:
#         img = Image.open(f).convert('L')  # convert to grayscale
    
#     # Normalize pixel values to [0, 1]
#     img_np = np.array(img, dtype=np.float32) / 255.0
    
#     # Convert back to a normalized PIL Image
#     img_normalized = Image.fromarray((img_np * 255).astype(np.uint8))
    
#     return img_normalized


from PIL import Image, ImageEnhance

def enhance_image(
    image_pil, 
    brightness_factor=2.0, 
    contrast_factor=1.3, 
    sharpness_factor=1.5, 
    color_factor=2.0
):
    """
    Apply image enhancements: brightness, contrast, sharpness, and color.

    Args:
        image_pil (PIL.Image.Image): Input PIL image.
        brightness_factor (float): Factor for brightness enhancement.
        contrast_factor (float): Factor for contrast enhancement.
        sharpness_factor (float): Factor for sharpness enhancement.
        color_factor (float): Factor for color enhancement.

    Returns:
        PIL.Image.Image: Enhanced image.
    """
    enhanced = image_pil.copy()

    # Brightness
    enhancer = ImageEnhance.Brightness(enhanced)
    enhanced = enhancer.enhance(brightness_factor)

    # Contrast
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(contrast_factor)

    # Sharpness
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(sharpness_factor)

    # Color
    enhancer = ImageEnhance.Color(enhanced)
    enhanced = enhancer.enhance(color_factor)

    return enhanced
