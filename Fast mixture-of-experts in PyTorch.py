import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import random

import kornia
import kornia.augmentation as K

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
torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Assuming you're using a CUDA-capable GPU
device = torch.device(device)
print(device)


class RandomNoise(torch.nn.Module):
    """Applies random Gaussian noise to each channel of a color image."""
    def __init__(self, mean=0.0, std=0.1, p=0.5):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(x) * self.std + self.mean
            x = x + noise
            x = torch.clamp(x, 0.0, 1.0)  # Ensures values remain within [0, 1] range for normalized images
        return x

class MorphologyEx(torch.nn.Module):
    """Applies random morphological operations: open, close, top hat."""
    def __init__(self, kernel_size=(3, 3), operation="open", p=0.5):
        super().__init__()
        self.kernel = torch.ones(1, 1, *kernel_size, device=device)  # Single-channel kernel but will work for each channel
        self.operation = operation
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() < self.p:
            if self.operation == "open":
                x = kornia.morphology.opening(x, self.kernel.repeat(x.shape[1], 1, 1, 1))
            elif self.operation == "close":
                x = kornia.morphology.closing(x, self.kernel.repeat(x.shape[1], 1, 1, 1))
            elif self.operation == "tophat":
                x = kornia.morphology.top_hat(x, self.kernel.repeat(x.shape[1], 1, 1, 1))
        return x

class Erosion(torch.nn.Module):
    """Applies random erosion."""
    def __init__(self, kernel_size=(3, 3), p=0.5):
        super().__init__()
        self.kernel = torch.ones(1, 1, *kernel_size, device=device)  # Single-channel kernel
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() < self.p:
            x = kornia.morphology.erosion(x, self.kernel.repeat(x.shape[1], 1, 1, 1))
        return x

class Dilation(torch.nn.Module):
    """Applies random dilation."""
    def __init__(self, kernel_size=(3, 3), p=0.5):
        super().__init__()
        self.kernel = torch.ones(1, 1, *kernel_size, device=device)  # Single-channel kernel
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() < self.p:
            x = kornia.morphology.dilation(x, self.kernel.repeat(x.shape[1], 1, 1, 1))
        return x
class CustomDataset(Dataset):
    def __init__(self, dataset, label_offset):
        self.dataset = dataset
        self.label_offset = label_offset

    def __getitem__(self, index):
        # Retrieve the image and label from the original dataset
        img, label = self.dataset[index]

        # Adjust the label by adding the offset
        adjusted_label = label + self.label_offset

        return img, adjusted_label

    def __len__(self):
        return len(self.dataset)


# Define transformations
resize_and_to_tensor = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert to 3 channels
])

to_tensor = transforms.Compose([
    transforms.ToTensor()
])

# Load and wrap datasets with appropriate transformations and label adjustments
mnist_train = CustomDataset(datasets.MNIST(root='./data', train=True, download=True,
                                                        transform=resize_and_to_tensor), label_offset=0)
mnist_test = CustomDataset(datasets.MNIST(root='./data', train=False, download=True,
                                                        transform=resize_and_to_tensor), label_offset=0)

fashion_mnist_train = CustomDataset(datasets.FashionMNIST(root='./data', train=True, download=True,
                                                        transform=resize_and_to_tensor), label_offset=10)
fashion_mnist_test = CustomDataset(datasets.FashionMNIST(root='./data', train=False, download=True,
                                                        transform=resize_and_to_tensor), label_offset=10)

SVHN_train = datasets.SVHN(root='./data', download=True,
                                                        transform=to_tensor)
# Filter the dataset to only include the selected classes
def filter_by_class(dataset, classes):
    class_indices = [i for i in range(len(dataset)) if int(dataset[i][1]) in classes]
    return Subset(dataset, class_indices)

SVHN_train = filter_by_class(SVHN_train, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
SVHN_test = CustomDataset(Subset(SVHN_train,torch.arange(0,int(len(SVHN_train)*0.9))), label_offset=20)
SVHN_train = CustomDataset(Subset(SVHN_train,torch.arange(int(len(SVHN_train)*0.9),len(SVHN_train))), label_offset=20)

cifar10_train = CustomDataset(datasets.CIFAR10(root='./data', train=True, download=True,
                                                        transform=to_tensor), label_offset=20)
cifar10_test = CustomDataset(datasets.CIFAR10(root='./data', train=False, download=True,
                                                        transform=to_tensor), label_offset=20)

def randomsub(dataset,num):
    return Subset(dataset,torch.randint(0,len(dataset),(num,)))

train_num = 128 #*128 *2
test_num = 1024 *4

# Combine datasets
combined_train_dataset = ConcatDataset(list(map(lambda x:randomsub(x,train_num),
                                            [mnist_train, fashion_mnist_train, SVHN_train,])))
combined_test_dataset  = ConcatDataset(list(map(lambda x:randomsub(x,test_num),
                                            [mnist_test, fashion_mnist_test, SVHN_test,])))

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=30):  # Adjusted for 30 combined classes
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 120)
        self.fc_bn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc_bn2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, num_classes)
        self.criterion=nn.CrossEntropyLoss()
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(F.relu(self.fc_bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.fc_bn2(self.fc2(x))))
        x = self.fc3(x)
        return x
    
    def get_loss(self, outputs, inputs, labels):
        return self.criterion(outputs,labels)

class SimpleGate(nn.Module):
    def __init__(self, num_classes=3):  # Adjusted for 3 combined classes
        super(SimpleGate, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 120)
        self.fc_bn1 = nn.BatchNorm1d(120)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84)
        self.fc_bn2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc_bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.fc_bn2(self.fc2(x))))
        x = self.fc3(x)
        return x
    
class SimpleVAE(nn.Module):
    def __init__(self, input_dim=32*32*3, hidden_dims=[256, 128], latent_dim=20, gate_dim=3, dropout_rate=0.5):
        super(SimpleVAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_mean = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)
        
        self.fc_gate = nn.Linear(latent_dim, gate_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dims[1])
        self.bn3 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.bn4 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(hidden_dims[0], input_dim)

    def gate(self, x):
        # Use the encoder as the gate function
        mu, _ = self.encode(x.view(-1, 32*32*3))
        return mu # Using just the mean as the gate's output

    def lock_encoder(self):
        # Lock weights of the encoder
        for params in [n.parameters() for n in [self.fc1,self.bn1,self.fc2,self.bn2,self.fc_mean,self.fc_logvar]]:
            for param in params:
                param.requires_grad = False

    def encode(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout1(h)
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.dropout2(h)
        return self.fc_mean(h), self.fc_logvar(h)  # Returns mean and log variance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Random noise
        return mu + eps * std  # This is the latent vector

    def decode(self, z):
        h = F.relu(self.bn3(self.fc3(z)))
        h = self.dropout3(h)
        h = F.relu(self.bn4(self.fc4(h)))
        h = self.dropout4(h)
        return torch.sigmoid(self.fc5(h))  # Reconstruction

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 32*32*3))  # Flatten the input
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar    

    # Loss function for VAE
    def get_loss(self, outputs, inputs, labels):
        recon_x, mu, logvar = outputs
        # Reconstruction loss (MSE loss or Binary Cross-Entropy loss can be used)
        recon_loss = F.mse_loss(recon_x, inputs.view(-1, 32*32*3), reduction='sum')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        return recon_loss + kl_loss
(
    
# gate_model = SimpleGate(num_classes=3).to(device)
# # Initialize the expert models
# mnist_model = SimpleCNN(num_classes=30).to(device)
# SVHN_model = SimpleCNN(num_classes=30).to(device)
# fashion_mnist_model = SimpleCNN(num_classes=30).to(device)
# # Define the expert models list
# expert_models = [mnist_model, fashion_mnist_model, SVHN_model]

# # Use DataLoader to handle batching
# train_loader = DataLoader(combined_train_dataset, batch_size=1024, shuffle=True)
# test_loader = DataLoader(combined_test_dataset, batch_size=1024, shuffle=False)

# def test_accuracy(gate_model, expert_models, test_loader):
#     correct = 0
#     total = 0
#     with torch.no_grad():  # No need to calculate gradients during testing
#         for data in test_loader:
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
            
#             # Forward pass through the gate model to select the expert
#             gate_predictions = torch.argmax(gate_model(inputs), dim=1)
            
#             # Iterate through the experts based on the gate's decision
#             for idx, expert_model in enumerate(expert_models):
#                 expert_mask = gate_predictions == idx
#                 # print(idx, expert_mask.sum())
#                 if expert_mask.sum() > 0:
#                     expert_inputs = inputs[expert_mask]
#                     expert_labels = labels[expert_mask]

#                     expert_outputs =expert_model(expert_inputs)
#                     pred = expert_outputs.argmax(dim=1, keepdim=True)

#                     total += expert_labels.size(0)
#                     correct += pred.eq(expert_labels.view_as(pred)).sum().item()
    
#     accuracy = 100 * correct / total
#     return accuracy

# # Assuming `test_loader` is defined and contains the combined test dataset
# test_acc = test_accuracy(gate_model, expert_models, test_loader)
# print(f'Accuracy of the network on the test images: {test_acc:.2f}%')

# # Define loss function and optimizers
# criterion = nn.CrossEntropyLoss()
# optimizer_gate = optim.Adam(gate_model.parameters(), lr=0.001)#optim.SGD(gate_model.parameters(), lr=0.001, momentum=0.9) #optim.Adam(gate_model.parameters(), lr=0.0001)
# optimizer_experts = [optim.Adam(model.parameters(), lr=0.001) for model in expert_models]

# # Assuming `train_loader` is defined and contains the combined dataset
# num_epochs = 100
# for epoch in range(num_epochs):  # Loop over the dataset multiple times
#     gate_model.running_loss=0.0
#     mnist_model.running_loss=0.0
#     SVHN_model.running_loss=0.0
#     fashion_mnist_model.running_loss=0.0
#     for i, data in enumerate(train_loader, 0):
#         # Zero the parameter gradients
#         for optimizer in [optimizer_gate] + optimizer_experts:
#             optimizer.zero_grad()

#         # Forward pass through the gate
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)

#         gate_outputs = torch.softmax(gate_model(inputs), dim=1)
#         loss = criterion(gate_outputs, labels // 10)  # Simplified for demonstration; may need adjustment
#         # loss.backward()
#         # optimizer_gate.step()
#         gate_model.running_loss += loss.item()

#         gate_predictions = torch.argmax(gate_outputs, dim=1)
#         # Forward and backward passes through the selected expert models based on gate's decision
#         for idx, expert_model in enumerate(expert_models):
#             expert_mask = gate_predictions == idx
#             # print(expert_mask.sum())
#             if expert_mask.sum() > 0:
#                 expert_inputs = inputs[expert_mask]
#                 expert_labels = labels[expert_mask]

#                 expert_outputs = expert_model(expert_inputs)
#                 expert_outputs = (torch.softmax(gate_outputs[expert_mask], dim=1).reshape(-1,3,1) * expert_outputs.reshape(-1,3,10)).reshape(-1,30)
#                 expert_outputs = torch.softmax(expert_outputs, dim=1)

#                 loss = criterion(expert_outputs, expert_labels)
#                 loss.backward()
#                 optimizer_experts[idx].step()
#                 expert_model.running_loss += loss.item()

#     running_loss = {    'gate_model':gate_model.running_loss / len(train_loader),
#                         'mnist_model':mnist_model.running_loss / len(train_loader),
#                         'SVHN_model':SVHN_model.running_loss / len(train_loader),
#                         'fashion_mnist_model':fashion_mnist_model.running_loss / len(train_loader),
#                     }
#     print(f"Epoch {epoch + 1}, Loss: { running_loss }")
#     test_acc = test_accuracy(gate_model, expert_models, test_loader)
#     print(f'Accuracy of the network on the test images: {test_acc:.2f}%')

# print('Finished Training')

# class MixtureOfExperts(nn.Module):
#     def __init__(self,gate_model=None,expert_models=None):
#         super(MixtureOfExperts, self).__init__()
#         # self.gate = SimpleGate(num_classes=3).to(device)  # 3 experts
#         self.gate = SimpleVAE(latent_dim=3).to(device)  # 3 experts
#         self.expert_models = [SimpleCNN(num_classes=30).to(device),SimpleCNN(num_classes=30).to(device),SimpleCNN(num_classes=30).to(device)]#expert_models
#         # self.mnist_expert = SimpleCNN(num_classes=30)
#         # self.fashion_mnist_expert = SimpleCNN(num_classes=30)
#         # self.cifar10_expert = SimpleCNN(num_classes=30)

#     def forward(self, x):
#         # Forward pass through the gate model to select the expert
#         gate_outputs = self.gate.gate(x)
#         gating_weights = F.softmax(gate_outputs, dim=1) # (batch_size, num_experts)
#         gate_predictions = torch.argmax(gate_outputs, dim=1)

#         # Initialize a tensor to store the outputs from the expert models
#         expert_outputs = torch.zeros(x.size(0), 3, 30).to(device)  # Assuming 30 total classes

#         # Iterate through each expert model and select the output based on the gate's decision
#         for idx, expert in enumerate(self.expert_models):
#             expert_mask = gate_predictions == idx
#             if expert_mask.sum() > 0:
#                 expert_inputs = x[expert_mask]
#                 # Only perform forward pass on the selected inputs for this expert
#                 outputs = expert(expert_inputs)
#                 # Place the outputs in the corresponding positions of the `expert_outputs` tensor
#                 expert_outputs[expert_mask,idx,:] = outputs

#         expert_outputs = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)
#         return gate_outputs,expert_outputs
    
#     def predict(self, x):
#         with torch.no_grad():  # No need to calculate gradients during testing
#             _,x = self.forward(x)
#             y = torch.argmax(x, dim=1)
#             return y

#     def train_model(self, train_loader, optimizer, criterion, epochs, device):
#         self.train()  # Set the model to training mode

#         for epoch in range(epochs):
#             running_loss = 0.0
#             for inputs, labels in train_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)

#                 optimizer.zero_grad()  # Zero the parameter gradients

#                 gating_weights = F.softmax(self.gate.gate(inputs), dim=1) # (batch_size, num_experts)
#                 # print('gating_weights', gating_weights.shape)

#                 #########################################################
#                 expert_outputs = torch.stack([expert(inputs) for expert in self.expert_models], dim=1) # (batch_size, num_experts, embed_size)
#                 output = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)
#                 # print('gating_weights.unsqueeze(-1)', gating_weights.unsqueeze(-1).shape)          
#                 # print('output', output.shape)
#                 loss = criterion(output, labels)
#                 ###########################################################
#                 # output = torch.zeros(inputs.size(0), 3, 30).to(device)  # Assuming 30 total classes
#                 # # Iterate through each expert model and select the output based on the gate's decision
#                 # for idx, expert in enumerate(self.expert_models):
#                 #     expert_mask = torch.argmax(gating_weights, dim=1) == idx
#                 #     if expert_mask.sum() > 0:
#                 #         expert_inputs = inputs[expert_mask]
#                 #         # Only perform forward pass on the selected inputs for this expert
#                 #         outputs = expert(expert_inputs)
#                 #         # Place the outputs in the corresponding positions of the `output` tensor
#                 #         output[expert_mask,idx,:] = outputs

#                 # # Compute loss
#                 # loss = criterion(torch.max(output,dim=1).values, labels)
#                 ###########################################################
                
#                 # Backward pass and optimize
#                 loss.backward()
#                 optimizer.step()

#                 running_loss += loss.item()

#             epoch_loss = running_loss / len(train_loader)            
#             yield f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}'

#         print('Training complete')

#     def test_accuracy(self, test_loader):
#         correct = 0
#         total = 0
#         with torch.no_grad():  # No need to calculate gradients during testing
#             for data in test_loader:
#                 inputs, labels = data
#                 inputs, labels = inputs.to(device), labels.to(device)

#                 pred = self.predict(inputs)
#                 total += labels.size(0)
#                 correct += pred.eq(labels.view_as(pred)).sum().item()
        
#         accuracy = 100 * correct / total
#         return accuracy
# # moe = MixtureOfExperts(gate_model,expert_models)
# moe = MixtureOfExperts().to(device)
)

class LinearMoE(nn.Module):
    def __init__(self, num_experts, input_size, output_size, bias=True):
        super(LinearMoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(input_size, output_size, bias=bias) for _ in range(num_experts)])
        # Gating network with a softmax output
        self.gate = nn.Linear(input_size, num_experts, bias=bias)
        self.input_size, self.output_size = input_size, output_size

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1,x_shape[-1])
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gating_weights = F.softmax(self.gate(x), dim=1)

        # Weighted sum of expert outputs
        output = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)
        output = output.reshape(*[*x_shape[:-1],self.output_size])

        return output
    
class ConvMoE(nn.Module):
    def __init__(self, num_experts, in_channels, out_channels, kernel_size, stride, padding=0):
        super(ConvMoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) for _ in range(num_experts)])
        self.gate = nn.Conv2d(in_channels, num_experts, kernel_size, stride, padding)

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # Shape: [batch, num_experts, channels, height, width]
        gating_weights = F.softmax(self.gate(x), dim=1)  # Shape: [batch, num_experts, height, width]
        gating_weights = gating_weights.unsqueeze(2)  # Add channel dimension
        output = torch.sum(gating_weights * expert_outputs, dim=1)
        return output

class SimpleMoECNN(nn.Module):
    def __init__(self, num_classes=32, num_experts=3, dropout_rate=0.5):
        super(SimpleMoECNN, self).__init__()
        self.num_experts = num_experts

        # Convolutional Layers with BatchNorm and Pooling
        self.conv_layers = nn.Sequential(
            ConvMoE(num_experts, 3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            ConvMoE(num_experts, 32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            LinearMoE(num_experts, 64 * 8 * 8, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            LinearMoE(num_experts, 120, num_classes)
        )

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

    def get_loss(self, outputs, inputs, labels):
        return self.criterion(outputs, labels)
    
# B -> Batch Size
# C -> Number of Input Channels
# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# E -> Embedding Dimension
# S -> Sequence Length = IH/P * IW/P
# Q -> Query Sequence length
# K -> Key Sequence length
# V -> Value Sequence length (same as Key length)
# H -> Number of heads
# HE -> Head Embedding Dimension = E/H
ConvMoE_3 = lambda n_channels,embed_dim,patch_size:  ConvMoE(3, n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
LinearMoE_3 = lambda input_size,output_size:  LinearMoE(3, input_size, output_size, bias=True)

class EmbedLayer(nn.Module):
    def __init__(self, n_channels, embed_dim, image_size, patch_size):
        super().__init__()
        self.conv1 = ConvMoE(3, n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)# nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)  # Pixel Encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)  # Cls Token
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, embed_dim), requires_grad=True)  # Positional Embedding

    def forward(self, x):
        x = self.conv1(x)  # B C IH IW -> B E IH/P IW/P (Embedding the patches)
        x = x.reshape([x.shape[0], x.shape[1], -1])  # B E IH/P IW/P -> B E S (Flattening the patches)
        x = x.transpose(1, 2)  # B E S -> B S E 
        # print('EmbedLayer transpose',x.shape)
        x = torch.cat((torch.repeat_interleave(self.cls_token, x.shape[0], 0), x), dim=1)  # Adding classification token at the start of every sequence
        x = x + self.pos_embedding  # Adding positional embedding
        # print('EmbedLayer x',x.shape)
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_attention_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_attention_heads = n_attention_heads
        self.head_embed_dim = embed_dim // n_attention_heads
        
        self.queries = LinearMoE_3( self.embed_dim, self.head_embed_dim * self.n_attention_heads)#nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.keys    = LinearMoE_3( self.embed_dim, self.head_embed_dim * self.n_attention_heads)#nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.values  = LinearMoE_3( self.embed_dim, self.head_embed_dim * self.n_attention_heads)#nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)

    def forward(self, x):
        m, s, e = x.shape
        # print('x.shape',x.shape)
        xq = self.queries(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  # B, Q, E -> B, Q, H, HE
        xq = xq.transpose(1, 2)  # B, Q, H, HE -> B, H, Q, HE
        # print('xq.shape',xq.shape)
        xk = self.keys(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  # B, K, E -> B, K, H, HE
        xk = xk.transpose(1, 2)  # B, K, H, HE -> B, H, K, HE
        # print('xk.shape',xq.shape)
        xv = self.values(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim)  # B, V, E -> B, V, H, HE
        xv = xv.transpose(1, 2)  # B, V, H, HE -> B, H, V, HE
        # print('xv.shape',xq.shape)

        # Compute Attention Matrix
        xk = xk.transpose(-1, -2)  # B, H, K, HE -> B, H, HE, K
        x_attention = torch.matmul(xq, xk)  # B, H, Q, HE  *  B, H, HE, K -> B, H, Q, K

        x_attention /= float(self.head_embed_dim) ** 0.5
        x_attention = torch.softmax(x_attention, dim=-1)

        # Compute Attention Values
        x = torch.matmul(x_attention, xv)  # B, H, Q, K * B, H, V, HE -> B, H, Q, HE

        # Format the output
        x = x.transpose(1, 2)  # B, H, Q, HE -> B, Q, H, HE
        x = x.reshape(m, s, e)  # B, Q, H, HE -> B, Q, E
        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim, n_attention_heads, forward_mul):
        super().__init__()
        self.attention = SelfAttention(embed_dim, n_attention_heads)
        self.fc1 = LinearMoE_3( embed_dim, embed_dim * forward_mul)#nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.GELU()
        self.fc2 = LinearMoE_3( embed_dim * forward_mul, embed_dim)#nn.Linear(embed_dim * forward_mul, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x)) # Skip connections
        x = x + self.fc2(self.activation(self.fc1(self.norm2(x))))  # Skip connections
        return x

class Classifier(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        # Newer architectures skip fc1 and activations and directly apply fc2.
        self.fc1 = LinearMoE_3( embed_dim, embed_dim)#nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = LinearMoE_3( embed_dim, n_classes)#nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = x[:, 0, :]  # Get CLS token
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class VisionMoETransformer(nn.Module):
    def __init__(self, n_channels=3, embed_dim=64, n_layers=6, n_attention_heads=4,
                 forward_mul=2, image_size=32, patch_size=4, n_classes=32):
        super().__init__()
        self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size)
        self.encoder = nn.Sequential(*[Encoder(embed_dim, n_attention_heads, forward_mul
                                               ) for _ in range(n_layers)], nn.LayerNorm(embed_dim))
        self.norm = nn.LayerNorm(embed_dim) # Final normalization layer after the last block
        self.classifier = Classifier(embed_dim, n_classes)
        self.criterion=nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embedding(x)
        # print('embedding',x.shape)
        x = self.encoder(x)
        # print('encoder',x.shape)
        x = self.norm(x)
        x = self.classifier(x)
        return x
        
    def get_loss(self, outputs, inputs, labels):
        return self.criterion(outputs,labels)

def simple_train(model,train_set,test_set,epochs=5,batch_size=64,augmentation=False):
    # Define the loss function and optimizer
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # Apply augmentations with Kornia
    if augmentation:
        augmentations = torch.nn.Sequential(
            K.RandomRotation(degrees=30.0, p=0.5),  # 70% chance to apply rotation
            K.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(5, 5), p=0.5),  # 50% chance
            K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.5),  # 40% chance to apply blur
            # K.RandomHorizontalFlip(p=0.6),  # 60% chance to apply horizontal flip
            K.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.5),  # 30% chance

            # New augmentations
            # MorphologyEx(kernel_size=(3, 3), operation="close", p=0.5),
            # Erosion(kernel_size=(3, 3), p=0.5),
            RandomNoise(mean=0.0, std=0.1, p=0.5),
            # MorphologyEx(kernel_size=(3, 3), operation="tophat", p=0.5),
            # Dilation(kernel_size=(3, 3), p=0.5),
            # MorphologyEx(kernel_size=(3, 3), operation="open", p=0.5),
            
        ).to(device)

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)


    def show_samples(dataset, num_samples=64):
        # Get a batch of samples
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
        images, labels = next(iter(data_loader))

        for i in range(len(images)):
            images[i] = augmentations(images[i])[0]
        # Denormalize if the dataset is normalized (adjust if necessary)
        # images = images / 2 + 0.5  # Assuming a [-1, 1] range normalization (common practice)

        grid_img = torchvision.utils.make_grid(images, nrow=int(num_samples**0.5))
        plt.figure(figsize=(15, 5))
        plt.imshow(grid_img.permute(1, 2, 0).numpy())  # Convert channels-first (C, H, W) to channels-last (H, W, C)
        plt.axis('off')
        plt.title("Sample Images from Dataset")
        plt.show()

    # Assuming you have a dataset variable defined, e.g., `train_dataset`
    show_samples(train_set)

    # Train the model
    def train(model, device, train_loader, optimizer, epochs=epochs):
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs, labels = inputs.to(device), labels.to(device)
                
                if augmentation: inputs = augmentations(inputs)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = model.get_loss(outputs, inputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 10==0:
                    print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss}')                
                    test(model, device, test_loader)
                    running_loss = 0.0

    # Evaluate the model
    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += model.get_loss(outputs, inputs, labels).item()
                if type(outputs) is not tuple:
                    pred = outputs.argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')

    model.to(device)

    # Run the training and testing
    train(model, device, train_loader, optimizer)

# gate_model = moe.gate
# mnist_model,fashion_mnist_model,SVHN_model = moe.expert_models
# simple_train(gate_model,combined_train_dataset,combined_test_dataset,2,lambel_func=lambda x:x//10)
# simple_train(mnist_model,mnist_train,mnist_test,2)
# simple_train(fashion_mnist_model,fashion_mnist_train,fashion_mnist_test,2)
# simple_train(SVHN_model,SVHN_train,SVHN_test,2)

# simple_train(gate_model,            combined_train_dataset,combined_test_dataset,5)
# simple_train(mnist_model,           combined_train_dataset,combined_test_dataset,1)
# simple_train(fashion_mnist_model,   combined_train_dataset,combined_test_dataset,1)
# simple_train(SVHN_model,            combined_train_dataset,combined_test_dataset,1)

svit = VisionMoETransformer(embed_dim=64, n_layers=6, image_size=32, n_classes=32)
moe = SimpleMoECNN(num_classes=32)

simple_train(moe,            combined_train_dataset,combined_test_dataset,300,32, True)
# simple_train(moe,            combined_train_dataset,fashion_mnist_test,1)
# simple_train(moe,            combined_train_dataset,SVHN_test,1)

# Use DataLoader to handle batching
train_loader = DataLoader(combined_train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(combined_test_dataset, batch_size=512, shuffle=False)
for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    images, labels = inputs.cpu(), labels.cpu()
    break

# Show images and labels
def imshow(img):
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

f,t = 128,128+16

# Display images
imshow(torchvision.utils.make_grid(images[f:t]))
# Print labels
print('    Raw Labels: ', labels[f:t].tolist())
print('Predict Labels: ', moe(images[f:t].to(device)).cpu().detach().numpy().argmax(1).tolist())

moe(images[f:t].to(device)).cpu().detach().numpy().argmax(1)

# for params in [n.parameters() for n in [gate_model.fc1,gate_model.bn1,gate_model.fc2,gate_model.bn2,gate_model.fc_mean,gate_model.fc_logvar]]:
#     for param in params:
#         param.requires_grad = False
        
# print(f'Accuracy of the network on the test images: {moe.test_accuracy(test_loader):.2f}%')
# for p in moe.train_model(train_loader,optim.Adam(moe.parameters(), lr=0.001),nn.CrossEntropyLoss(),100,device):
#     print(p)
#     print(f'Accuracy of the network on the test images: {moe.test_accuracy(test_loader):.2f}%')