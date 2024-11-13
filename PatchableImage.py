# Correcting the merging process to ensure patch dimensions match the original image.

class PatchableImage:
    def __init__(self, img_array, patch_size=(512, 512)):
        """
        Initialize the PatchableImage with a numpy array and patch size.
        """
        self.patch_size = patch_size
        self.img_raw = img_array
        self.img_raw_shape = img_array.shape[:2]  # Only height and width
        assert len(self.img_raw.shape) == 3 or len(self.img_raw.shape) == 2

        # Padding image dimensions to be divisible by patch size
        pad_height = (patch_size[0] - self.img_raw_shape[0] % patch_size[0]) % patch_size[0]
        pad_width = (patch_size[1] - self.img_raw_shape[1] % patch_size[1]) % patch_size[1]
        self.img = self._add_margin(self.img_raw, pad_height, pad_width)

        # Extract patches
        self.patches, self.patches_merge_func = self.split_image_into_patches(self.img, 
                                                                              self.img.shape[0] // patch_size[0], 
                                                                              self.img.shape[1] // patch_size[1])

    def get_merge_imgs(self, imgs=None):
        """
        Merges patches into the original image dimensions.
        """
        if imgs is None:
            imgs = self.patches
        res = self.patches_merge_func(imgs)
        return res[:self.img_raw_shape[0], :self.img_raw_shape[1]]

    def _add_margin(self, img_array, pad_height, pad_width, color=0):
        """
        Add margins to the image to make it divisible by the patch size.
        """
        if len(img_array.shape) == 3:  # Color image
            pad = ((0, pad_height), (0, pad_width), (0, 0))
        else:  # Grayscale image
            pad = ((0, pad_height), (0, pad_width))

        return np.pad(img_array, pad, mode='constant', constant_values=color)

    def merge_patches(self, patches, rows, cols):
        """
        Merges patches into a single image.
        """
        merged_image = np.zeros((rows * self.patch_size[0], cols * self.patch_size[1], patches[0].shape[2]), dtype=patches[0].dtype)
        patch_idx = 0
        for i in range(rows):
            for j in range(cols):
                merged_image[i*self.patch_size[0]:(i+1)*self.patch_size[0], j*self.patch_size[1]:(j+1)*self.patch_size[1]] = patches[patch_idx]
                patch_idx += 1
        return merged_image

    def split_image_into_patches(self, image, n, m):
        """
        Splits an image into patches.
        """
        patch_h, patch_w = self.patch_size
        patches:list[np.ndarray] = [image[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                   for i in range(n) for j in range(m)]
        return patches, lambda x: self.merge_patches(x, rows=n, cols=m)
    
    def add_frame_to_patches(self, frame_color=(255, 255, 255), thickness=5):
        """
        Adds a white frame to each patch for debugging purposes.

        Parameters:
        frame_color (tuple): Color of the frame in RGB.
        thickness (int): Thickness of the frame.
        """
        framed_patches = []
        for patch in self.patches:
            framed_patch = patch.copy()
            framed_patch[:thickness, :, :] = frame_color  # Top border
            framed_patch[-thickness:, :, :] = frame_color  # Bottom border
            framed_patch[:, :thickness, :] = frame_color  # Left border
            framed_patch[:, -thickness:, :] = frame_color  # Right border
            framed_patches.append(framed_patch)
        return framed_patches

class PatchableImage:
    def __init__(self, img_tensor, patch_size=(512, 512), device='cuda'):
        """
        Initialize the PatchableImage with a PyTorch tensor, patch size, and device.
        """
        self.device = device
        self.patch_size = patch_size
        self.img_raw = img_tensor.to(self.device)  # Move input tensor to the specified device
        self.img_raw_shape = self.img_raw.shape[:2]  # Only height and width
        assert len(self.img_raw.shape) == 3 or len(self.img_raw.shape) == 4  # Ensure valid tensor shape (C, H, W) or (H, W)

        # Padding image dimensions to be divisible by patch size
        pad_height = (patch_size[0] - self.img_raw_shape[0] % patch_size[0]) % patch_size[0]
        pad_width = (patch_size[1] - self.img_raw_shape[1] % patch_size[1]) % patch_size[1]
        # self.img = self._add_margin(self.img_raw, pad_height, pad_width)

        # # # Extract patches
        # self.patches, self.patches_merge_func = self.split_image_into_patches(
        #     self.img,
        #     self.img.shape[-2] // patch_size[0],
        #     self.img.shape[-1] // patch_size[1]
        # )

    def get_merge_imgs(self, imgs=None):
        """
        Merges patches into the original image dimensions.
        """
        if imgs is None:
            imgs = self.patches
        res = self.patches_merge_func(imgs)
        return res[:, :self.img_raw_shape[0], :self.img_raw_shape[1]]  # Cropping to original size

    def _add_margin(self, img_tensor, pad_height, pad_width, color=0):
        """
        Add margins to the image tensor to make it divisible by the patch size.
        """
        pad = (0, pad_width, 0, pad_height)
        return F.pad(img_tensor, pad, mode='constant', value=color)

    def merge_patches(self, patches, rows, cols):
        """
        Merges patches into a single image tensor.
        """
        merged_image = torch.zeros(
            (patches[0].shape[0], rows * self.patch_size[0], cols * self.patch_size[1]),
            dtype=patches[0].dtype,
            device=self.device  # Ensure merged image is on the correct device
        )
        patch_idx = 0
        for i in range(rows):
            for j in range(cols):
                merged_image[:, i*self.patch_size[0]:(i+1)*self.patch_size[0], j*self.patch_size[1]:(j+1)*self.patch_size[1]] = patches[patch_idx]
                patch_idx += 1
        return merged_image

    def split_image_into_patches(self, image, n, m):
        """
        Splits an image tensor into patches.
        """
        patch_h, patch_w = self.patch_size
        patches = [
            image[:, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            for i in range(n) for j in range(m)
        ]
        return patches, lambda x: self.merge_patches(x, rows=n, cols=m)

    def add_frame_to_patches(self, frame_color=(255, 255, 255), thickness=5):
        """
        Adds a frame to each patch for debugging purposes.

        Parameters:
        frame_color (tuple): Color of the frame in RGB.
        thickness (int): Thickness of the frame.
        """
        frame_color_tensor = torch.tensor(frame_color, dtype=self.patches[0].dtype, device=self.device)
        framed_patches = []
        for patch in self.patches:
            framed_patch = patch.clone()
            framed_patch[:, :thickness , :] = frame_color_tensor.unsqueeze(-1)  # Top border
            framed_patch[:, -thickness:, :] = frame_color_tensor.unsqueeze(-1)  # Bottom border
            framed_patch[:, :, :thickness ] = frame_color_tensor.unsqueeze(-1)  # Left border
            framed_patch[:, :, -thickness:] = frame_color_tensor.unsqueeze(-1)  # Right border
            framed_patches.append(framed_patch)
        return framed_patches

# Test again with the corrected merging logic
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sample_image = torch.from_numpy(
    np.asarray(Image.open(r'c:\Users\ixs-lbook02\OneDrive - 株式会社イクシス\画像\5k.jpg')).copy()
    )
patchable_img = PatchableImage(sample_image, patch_size=(1280, 1280))
# # framed_patches = patchable_img.add_frame_to_patches()
# # merged_image = patchable_img.get_merge_imgs(framed_patches)

# # Plot the original and merged images for comparison
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(sample_image)
# ax[0].set_title("Original Image")
# ax[0].axis('off')

# ax[1].imshow(merged_image)
# ax[1].set_title("Merged Image (After Splitting)")
# ax[1].axis('off')

# plt.show()
