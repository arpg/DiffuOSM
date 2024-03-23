import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb

# Add the helpers directory to the Python module search path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_extraction', 'helpers'))

from read_frames import read_bin_file

# Dataset class for preprocessing and loading data
class SceneCompletionDataset(Dataset):
    def __init__(self, seq, num_points):
        self.seq = seq
        self.num_points = num_points
        self.data_dir = os.environ['KITTI360_DATASET']  # Use the extracted directory directly
        self.scans = self.get_scan_numbers()

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx):
        scan_num = self.scans[idx]
        obs_edges_path = os.path.join(self.data_dir, f"{scan_num:010d}_obs_edges.bin")
        unobs_edges_path = os.path.join(self.data_dir, f"{scan_num:010d}_unobs_edges.bin")
        obs_points_path = os.path.join(self.data_dir, f"{scan_num:010d}_obs_points.bin")
        accum_points_path = os.path.join(self.data_dir, f"{scan_num:010d}_accum_points.bin")

        # Check if all required files exist
        if not (os.path.exists(obs_edges_path) and os.path.exists(unobs_edges_path) and
                os.path.exists(obs_points_path) and os.path.exists(accum_points_path)):
            # Skip this sample and return the next valid sample
            return self.__getitem__((idx + 1) % len(self))

        obs_edges = read_bin_file(obs_edges_path)
        unobs_edges = read_bin_file(unobs_edges_path)
        obs_points = read_bin_file(obs_points_path)
        accum_points = read_bin_file(accum_points_path)

        # Preprocessing steps
        obs_edges = self.preprocess_points(obs_edges)
        unobs_edges = self.preprocess_points(unobs_edges)
        obs_points = self.preprocess_points(obs_points)
        accum_points = self.preprocess_points(accum_points)

        # Point cloud subsampling
        obs_edges = self.subsample_points(obs_edges)
        unobs_edges = self.subsample_points(unobs_edges)
        obs_points = self.subsample_points(obs_points)
        accum_points = self.subsample_points(accum_points)

        return obs_edges, unobs_edges, obs_points, accum_points

    def preprocess_points(self, points):
        # Normalize coordinates to the range [-1, 1]
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        normalized_points = 2 * (points - min_coords) / (max_coords - min_coords) - 1
        return torch.tensor(normalized_points, dtype=torch.float32)

    def subsample_points(self, points):
        # Random subsampling of points
        idx = np.random.choice(points.shape[0], self.num_points, replace=False)
        subsampled_points = points[idx]
        return subsampled_points

    def get_scan_numbers(self):
        scan_files = [f for f in os.listdir(self.data_dir) if f.endswith("_accum_points.bin")]
        scan_numbers = []
        for file in scan_files:
            try:
                scan_num = int(file.split("_")[0])
                obs_edges_path = os.path.join(self.data_dir, f"{scan_num:010d}_obs_edges.bin")
                unobs_edges_path = os.path.join(self.data_dir, f"{scan_num:010d}_unobs_edges.bin")
                obs_points_path = os.path.join(self.data_dir, f"{scan_num:010d}_obs_points.bin")
                print(f"Checking scan number: {scan_num}")
                print(f"obs_edges_path: {obs_edges_path}")
                print(f"unobs_edges_path: {unobs_edges_path}")
                print(f"obs_points_path: {obs_points_path}")
                print(f"File exists: obs_edges: {os.path.exists(obs_edges_path)}, unobs_edges: {os.path.exists(unobs_edges_path)}, obs_points: {os.path.exists(obs_points_path)}")
                if os.path.exists(obs_edges_path) and os.path.exists(unobs_edges_path) and os.path.exists(obs_points_path):
                    scan_numbers.append(scan_num)
                    print(f"Added scan number: {scan_num}")
                else:
                    print(f"Skipping scan number: {scan_num}")
            except (ValueError, IndexError):
                print(f"Skipping file: {file}")
                continue
        return sorted(scan_numbers)

# Diffusion model architecture
class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DiffusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, obs_edges, unobs_edges, obs_points):
        # Concatenate input features
        x = torch.cat((obs_edges, unobs_edges, obs_points), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

# Training loop
def train(model, dataloader, optimizer, criterion, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        for batch_idx, data in enumerate(dataloader, 0):
            obs_edges, unobs_edges, obs_points, accum_points = data
            obs_edges = obs_edges.to(device)
            unobs_edges = unobs_edges.to(device)
            obs_points = obs_points.to(device)
            accum_points = accum_points.to(device)

            # Forward pass
            outputs = model(obs_edges, unobs_edges, obs_points)
            loss = criterion(outputs, accum_points)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # Log metrics to wandb
        wandb.log({"epoch": epoch+1, "loss": epoch_loss / len(dataloader)})

        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss / len(dataloader):.4f}")

# Hyperparameters
input_dim = 9  # Assumes 3D points (x, y, z) for each input
hidden_dim = 512
output_dim = 3
learning_rate = 0.001
batch_size = 32
num_epochs = 100
num_points = 1024  # Number of points to subsample

# Set the sequence number
seq = 0

# Initialize wandb
wandb.init(project="diffusom", config={
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "output_dim": output_dim,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "num_points": num_points
})

# Create dataset and dataloader
dataset = SceneCompletionDataset(seq, num_points)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionModel(input_dim, hidden_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Train the model
train(model, dataloader, optimizer, criterion, device, num_epochs)