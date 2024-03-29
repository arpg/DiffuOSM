import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
import open3d as o3d
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.distance import cdist
from pyemd import emd
from datetime import datetime

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

        # Check if any of the arrays are empty
        if obs_edges.size == 0 or unobs_edges.size == 0 or obs_points.size == 0 or accum_points.size == 0:
            # Skip this sample and return the next valid sample
            return self.__getitem__((idx + 1) % len(self))

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
        if points.size == 0:
            return torch.empty(0, 3, dtype=torch.float32)
        
        # Normalize coordinates to the range [-1, 1]
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        normalized_points = 2 * (points - min_coords) / (max_coords - min_coords + 1e-8) - 1
        return torch.tensor(normalized_points, dtype=torch.float32)
    
    def subsample_points(self, points):
        if isinstance(points, torch.Tensor):
            points = points.numpy()

        if points.size == 0:
            return torch.empty(0, 3, dtype=torch.float32)

        # Random subsampling of points
        if points.shape[0] >= self.num_points:
            idx = np.random.choice(points.shape[0], self.num_points, replace=False)
            subsampled_points = points[idx]
        else:
            # If the input points array has fewer points than the desired subsample size,
            # randomly duplicate points to reach the desired size
            num_pad_points = self.num_points - points.shape[0]
            pad_idx = np.random.choice(points.shape[0], num_pad_points, replace=True)
            subsampled_points = np.concatenate((points, points[pad_idx]), axis=0)

        return torch.from_numpy(subsampled_points).float()

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

# U-Net block with cross-attention
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        self.cross_attn = nn.MultiheadAttention(out_channels, num_heads=4)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, t, cond):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        t_emb = self.time_emb(t)
        t_emb = t_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
        x = x + t_emb

        attn_output, _ = self.cross_attn(x, cond, cond)
        x = x + attn_output

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x

# Diffusion model architecture
class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_points, num_timesteps):
        super(DiffusionModel, self).__init__()
        self.num_points = num_points
        self.num_timesteps = num_timesteps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.down1 = UNetBlock(3, hidden_dim, hidden_dim)
        self.down2 = UNetBlock(hidden_dim, hidden_dim * 2, hidden_dim)
        self.down3 = UNetBlock(hidden_dim * 2, hidden_dim * 4, hidden_dim)

        self.bottleneck = UNetBlock(hidden_dim * 4, hidden_dim * 8, hidden_dim)

        self.up1 = UNetBlock(hidden_dim * 12, hidden_dim * 4, hidden_dim)
        self.up2 = UNetBlock(hidden_dim * 6, hidden_dim * 2, hidden_dim)
        self.up3 = UNetBlock(hidden_dim * 3, hidden_dim, hidden_dim)

        self.out = nn.Conv1d(hidden_dim, 3, kernel_size=1)

    def forward(self, obs_edges, unobs_edges, obs_points, noisy_accum_points, t):
        # Reshape input features
        obs_edges = obs_edges.view(-1, self.num_points, 3).permute(0, 2, 1)
        unobs_edges = unobs_edges.view(-1, self.num_points, 3).permute(0, 2, 1)
        obs_points = obs_points.view(-1, self.num_points, 3).permute(0, 2, 1)
        noisy_accum_points = noisy_accum_points.view(-1, self.num_points, 3).permute(0, 2, 1)

        # Concatenate input features
        cond = torch.cat((obs_edges, unobs_edges, obs_points), dim=1)
        cond = self.embedding(cond)

        # U-Net with cross-attention
        down1 = self.down1(noisy_accum_points, t, cond)
        down2 = self.down2(down1, t, cond)
        down3 = self.down3(down2, t, cond)

        bottleneck = self.bottleneck(down3, t, cond)

        up1 = self.up1(torch.cat((bottleneck, down3), dim=1), t, cond)
        up2 = self.up2(torch.cat((up1, down2), dim=1), t, cond)
        up3 = self.up3(torch.cat((up2, down1), dim=1), t, cond)

        out = self.out(up3)

        return out.permute(0, 2, 1)

# Training loop
def train(model, dataloader, optimizer, criterion, device, num_epochs, model_save_dir, num_timesteps, beta_start, beta_end):
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

            # Generate timesteps
            t = torch.randint(0, num_timesteps, (accum_points.shape[0],), device=device).long()
            t_embed = t.float().view(-1, 1) / num_timesteps

            # Compute noise schedule parameters
            beta_t = beta_start + (beta_end - beta_start) * t / num_timesteps
            alpha_t = 1 - beta_t
            alpha_bar_t = torch.cumprod(alpha_t, dim=0)

            # Sample noise
            noise = torch.randn_like(accum_points)

            # Compute noisy accumulation points
            noisy_accum_points = torch.sqrt(alpha_bar_t).view(-1, 1, 1) * accum_points + torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1) * noise

            # Forward pass
            predicted_noise = model(obs_edges, unobs_edges, obs_points, noisy_accum_points, t_embed)
            loss = criterion(predicted_noise, noise)

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

    # Save the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"diffusion_model_e{num_epochs}_h{hidden_dim}_p{num_points}_{timestamp}.pth"
    model_path = os.path.join(model_save_dir, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")

# Inference function
def infer(model, obs_edges, unobs_edges, obs_points, num_timesteps, beta_start, beta_end, device):
    model.eval()
    with torch.no_grad():
        # Reshape input features
        obs_edges = obs_edges.view(-1, num_points * 3)
        unobs_edges = unobs_edges.view(-1, num_points * 3)
        obs_points = obs_points.view(-1, num_points * 3)

        # Initialize the accumulation points with random noise
        accum_points = torch.randn((1, num_points, 3), device=device)

        # Iterative denoising process
        for t in range(num_timesteps - 1, -1, -1):
            t_embed = torch.tensor([t / num_timesteps], device=device).float().view(1, 1)
            predicted_noise = model(obs_edges, unobs_edges, obs_points, accum_points, t_embed)
            beta_t = beta_start + (beta_end - beta_start) * t / num_timesteps
            alpha_t = 1 - beta_t
            alpha_bar_t = torch.prod(torch.tensor([1 - beta_start + (beta_end - beta_start) * i / num_timesteps for i in range(t + 1)], device=device))
            
            # Convert alpha_t and alpha_bar_t to tensors
            alpha_t = torch.tensor(alpha_t, device=device)
            alpha_bar_t = torch.tensor(alpha_bar_t, device=device)
            
            accum_points = (accum_points - beta_t / torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_t)

        return accum_points

def visualize_examples(models, dataset, device, num_examples=10, ensemble=False):
    if not ensemble:
        models = [models]  # Convert single model to a list for consistency

    indices = np.random.choice(len(dataset), num_examples, replace=False)
    
    for idx in indices:
        obs_edges, unobs_edges, obs_points, accum_points = dataset[idx]
        obs_edges = obs_edges.unsqueeze(0).to(device)
        unobs_edges = unobs_edges.unsqueeze(0).to(device)
        obs_points = obs_points.unsqueeze(0).to(device)
        
        with torch.no_grad():
            t = torch.tensor([0.0], device=device).float().view(1, 1)  # Add this line to provide the timestep embedding
            outputs = []
            for model in models:
                model.eval()
                output = model(obs_edges, unobs_edges, obs_points, t)  # Pass 't' as an argument to the model
                outputs.append(output)
            
            if ensemble:
                output = torch.mean(torch.stack(outputs), dim=0)
            else:
                output = outputs[0]
        
        obs_edges_np = obs_edges.squeeze().cpu().numpy()
        unobs_edges_np = unobs_edges.squeeze().cpu().numpy()
        obs_points_np = obs_points.squeeze().cpu().numpy()
        accum_points_np = accum_points.numpy()
        output_np = output.squeeze().cpu().numpy()
        
        pcd_obs_edges = o3d.geometry.PointCloud()
        pcd_obs_edges.points = o3d.utility.Vector3dVector(obs_edges_np)
        pcd_obs_edges.paint_uniform_color([1, 0, 0])  # Red
        
        pcd_unobs_edges = o3d.geometry.PointCloud()
        pcd_unobs_edges.points = o3d.utility.Vector3dVector(unobs_edges_np)
        pcd_unobs_edges.paint_uniform_color([0, 1, 0])  # Green
        
        pcd_obs_points = o3d.geometry.PointCloud()
        pcd_obs_points.points = o3d.utility.Vector3dVector(obs_points_np)
        pcd_obs_points.paint_uniform_color([0, 0, 1])  # Blue
        
        pcd_accum_points = o3d.geometry.PointCloud()
        pcd_accum_points.points = o3d.utility.Vector3dVector(accum_points_np)
        pcd_accum_points.paint_uniform_color([1, 1, 0])  # Yellow
        
        pcd_output = o3d.geometry.PointCloud()
        pcd_output.points = o3d.utility.Vector3dVector(output_np)
        pcd_output.paint_uniform_color([0, 1, 1])  # Cyan
        
        # Create a visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Example {idx+1}")
        
        # Add point clouds to the visualization
        vis.add_geometry(pcd_obs_edges)
        vis.add_geometry(pcd_unobs_edges)
        vis.add_geometry(pcd_obs_points)
        vis.add_geometry(pcd_accum_points)
        vis.add_geometry(pcd_output)
        
        # Add legend text labels
        legend_obs_edges = o3d.geometry.TextGeometry("Observed Edges (Red)", font_size=12)
        legend_obs_edges.translate([0, 0.1, 0])
        vis.add_geometry(legend_obs_edges)
        
        legend_unobs_edges = o3d.geometry.TextGeometry("Unobserved Edges (Green)", font_size=12)
        legend_unobs_edges.translate([0, 0.05, 0])
        vis.add_geometry(legend_unobs_edges)
        
        legend_obs_points = o3d.geometry.TextGeometry("Observed Points (Blue)", font_size=12)
        legend_obs_points.translate([0, 0, 0])
        vis.add_geometry(legend_obs_points)
        
        legend_accum_points = o3d.geometry.TextGeometry("Accumulated Points (Yellow)", font_size=12)
        legend_accum_points.translate([0, -0.05, 0])
        vis.add_geometry(legend_accum_points)
        
        legend_output = o3d.geometry.TextGeometry("Predicted Output (Cyan)", font_size=12)
        legend_output.translate([0, -0.1, 0])
        vis.add_geometry(legend_output)
        
        # Add coordinate axes
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(axes)
        
        # Run the visualization
        vis.run()
        vis.destroy_window()


def compute_metrics(model, test_dataloader, device):
    model.eval()
    total_hausdorff_dist = 0
    total_emd = 0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            obs_edges, unobs_edges, obs_points, accum_points = data
            obs_edges = obs_edges.to(device)
            unobs_edges = unobs_edges.to(device)
            obs_points = obs_points.to(device)
            accum_points = accum_points.to(device)

            # Generate point clouds using the trained model
            generated_points = infer(model, obs_edges, unobs_edges, obs_points, num_timesteps, beta_start, beta_end, device)

            # Compute Hausdorff Distance
            generated_points_np = generated_points.squeeze().cpu().numpy()
            accum_points_np = accum_points.squeeze().cpu().numpy()
            hausdorff_distance = max(directed_hausdorff(generated_points_np, accum_points_np)[0],
                                     directed_hausdorff(accum_points_np, generated_points_np)[0])
            total_hausdorff_dist += hausdorff_distance

            # Compute Earth Mover's Distance
            distance_matrix = cdist(generated_points_np, accum_points_np)
            emd_distance = emd(np.ones(generated_points_np.shape[0]) / generated_points_np.shape[0],
                               np.ones(accum_points_np.shape[0]) / accum_points_np.shape[0],
                               distance_matrix)
            total_emd += emd_distance

            num_samples += 1

    avg_hausdorff_dist = total_hausdorff_dist / num_samples
    avg_emd = total_emd / num_samples

    print(f"Average Hausdorff Distance: {avg_hausdorff_dist:.4f}")
    print(f"Average Earth Mover's Distance: {avg_emd:.4f}")


# Hyperparameters
num_points = 2048  # Number of points to subsample
input_dim = num_points * 3 * 3  # Assumes 3D points (x, y, z) for each input and 3 input tensors
hidden_dim = 4096
output_dim = num_points * 3
learning_rate = 0.001
batch_size = 64
num_epochs = 30
num_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
use_ensemble = False  # Set to True for ensemble training, False for single model training
num_ensemble = 3  # Number of models in the ensemble (only used if use_ensemble is True)

# Set the sequence number
seq = 0

# Create a directory to store the saved models
model_save_dir = "saved_models"
os.makedirs(model_save_dir, exist_ok=True)

# Initialize wandb
wandb.init(project="diffusom", config={
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "output_dim": output_dim,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "num_points": num_points,
    "num_timesteps": num_timesteps,
    "beta_start": beta_start,
    "beta_end": beta_end,
    "use_ensemble": use_ensemble,
    "num_ensemble": num_ensemble
})

# Create dataset and dataloader
dataset = SceneCompletionDataset(seq, num_points)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if use_ensemble:
    models = []
    optimizers = []
    for i in range(num_ensemble):
        model = DiffusionModel(input_dim, hidden_dim, output_dim, num_points, num_timesteps).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        models.append(model)
        optimizers.append(optimizer)
else:
    model = DiffusionModel(input_dim, hidden_dim, output_dim, num_points, num_timesteps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

# Print the size of the network in parameters
if use_ensemble:
    num_params = sum(p.numel() for p in models[0].parameters())
    print(f"Number of parameters in each model: {num_params}")
else:
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

# Train the model(s)
if use_ensemble:
    for i in range(num_ensemble):
        print(f"Training model {i+1}/{num_ensemble}")
        train(models[i], dataloader, optimizers[i], criterion, device, num_epochs, model_save_dir, num_timesteps, beta_start, beta_end)
else:
    train(model, dataloader, optimizer, criterion, device, num_epochs, model_save_dir, num_timesteps, beta_start, beta_end)

# Create a test dataset and dataloader
test_dataset = SceneCompletionDataset(seq, num_points)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Compute metrics over the test set
if use_ensemble:
    for i in range(num_ensemble):
        print(f"Metrics for model {i+1}/{num_ensemble}:")
        compute_metrics(models[i], test_dataloader, device)
else:
    compute_metrics(model, test_dataloader, device)

# Visualize examples from the train set
if use_ensemble:
    visualize_examples(models, dataset, device, num_examples=10, ensemble=True)
else:
    visualize_examples(model, dataset, device, num_examples=10)

# Visualize examples from the test set
if use_ensemble:
    visualize_examples(models, test_dataset, device, num_examples=10, ensemble=True)
else:
    visualize_examples(model, test_dataset, device, num_examples=10)