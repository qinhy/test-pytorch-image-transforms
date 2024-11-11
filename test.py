import time
import numpy as np
import torch
import kornia
import kornia.augmentation as K
import torchvision.transforms as T
from PIL import Image
import torch
import random
import numpy as np

# Set the seed for random number generators
seed = 42
torch.manual_seed(seed)  # For PyTorch
torch.cuda.manual_seed(seed)  # For PyTorch on GPU
torch.cuda.manual_seed_all(seed)  # If you use multiple GPUs

# For numpy
np.random.seed(seed)

# For random module (Python's built-in random library)
random.seed(seed)

# Ensure that PyTorch operations are deterministic for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device("cpu")# if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a simple RGB image
width, height = 1000, 1000
image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
image = T.ToTensor()(image_array).unsqueeze(0).to(device)  # Shape: [B, C, H, W]

# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(30),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
# transformed_image = transform(image)

# Apply augmentations with Kornia
augmentations = torch.nn.Sequential(
    K.RandomRotation(degrees=30.0, p=0.7),  # 70% chance to apply rotation
    K.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(5, 5), p=0.5),  # 50% chance
    K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.4),  # 40% chance to apply blur
    K.RandomHorizontalFlip(p=0.6),  # 60% chance to apply horizontal flip
    K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3)  # 30% chance
).to(device)

# Apply transformations
n = time.time()
for i in range(1000):
    transformed_image = augmentations(image)
print(time.time()-n)