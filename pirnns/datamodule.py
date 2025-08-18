import lightning as L
from torch.utils.data import DataLoader
import torch
import math
import numpy as np
from torch.utils.data import TensorDataset, random_split


class PathIntegrationDataModule(L.LightningDataModule):
    def __init__(
        self,
        num_trajectories: int,
        batch_size: int,
        num_workers: int,
        train_val_split: float,
        start_time: float,
        end_time: float,
        num_time_steps: int,
        arena_size: float,
        mu_speed: float,
        sigma_speed: float,
        tau_vel: float,
        # Place cell parameters
        num_place_cells: int,
        place_cell_rf: float,
        surround_scale: float,
        DoG: bool,
        periodic: bool,
        place_cell_seed: int,
    ) -> None:
        super().__init__()
        self.num_trajectories = num_trajectories
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.start_time = start_time
        self.end_time = end_time
        self.num_time_steps = num_time_steps
        self.dt = (end_time - start_time) / num_time_steps

        self.arena_size = arena_size
        self.mu_speed = mu_speed
        self.sigma_speed = sigma_speed
        self.tau_vel = tau_vel

        # Place cell parameters
        self.num_place_cells = num_place_cells
        self.place_cell_rf = place_cell_rf
        self.surround_scale = surround_scale
        self.DoG = DoG
        self.periodic = periodic
        
        # Initialize place cell centers
        np.random.seed(place_cell_seed)
        centers_x = np.random.uniform(-arena_size/2, arena_size/2, (num_place_cells,))
        centers_y = np.random.uniform(-arena_size/2, arena_size/2, (num_place_cells,))
        self.place_cell_centers = torch.tensor(np.vstack([centers_x, centers_y]).T, dtype=torch.float32)
        
        self.softmax = torch.nn.Softmax(dim=-1)

    def get_place_cell_activations(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Compute place cell activations for given positions.
        
        Args:
            pos: Positions of shape [batch_size, sequence_length, 2]
            
        Returns:
            activations: Place cell activations [batch_size, sequence_length, num_place_cells]
        """
        # Move centers to same device as pos
        centers = self.place_cell_centers.to(pos.device)
        
        # Compute distances: pos is [B, T, 2], centers is [Np, 2]
        d = torch.abs(pos[:, :, None, :] - centers[None, None, ...])
        
        if self.periodic:
            dx = d[:, :, :, 0]
            dy = d[:, :, :, 1]
            dx = torch.minimum(dx, self.arena_size - dx)
            dy = torch.minimum(dy, self.arena_size - dy)
            d = torch.stack([dx, dy], dim=-1)
        
        norm2 = (d**2).sum(-1)  # [B, T, Np]
        
        # Compute place cell activations with softmax normalization
        outputs = self.softmax(-norm2 / (2 * self.place_cell_rf**2))
        
        if self.DoG:
            # Subtract surround (larger sigma)
            surround = self.softmax(-norm2 / (2 * self.surround_scale * self.place_cell_rf**2))
            outputs = outputs - surround
            
            # Shift and scale to [0,1]
            min_output, _ = outputs.min(-1, keepdim=True)
            outputs = outputs + torch.abs(min_output)
            outputs = outputs / outputs.sum(-1, keepdim=True)
        
        return outputs

    def decode_position_from_place_cells(self, activation: torch.Tensor, k: int = 3) -> torch.Tensor:
        """
        Decode position from place cell activations using top-k method.
        
        Args:
            activation: Place cell activations [batch_size, sequence_length, num_place_cells]
            k: Number of top cells to use for decoding
            
        Returns:
            positions: Decoded positions [batch_size, sequence_length, 2]
        """
        centers = self.place_cell_centers.to(activation.device)
        _, idxs = torch.topk(activation, k=k, dim=-1)  # [B, T, k]
        pred_pos = centers[idxs].mean(-2)  # [B, T, 2]
        return pred_pos

    def simulate_trajectories(
        self,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulates a batch of trajectories following the Ornstein-Uhlenbeck process.
        (Brownian motion with a drift term).

        Parameters
        ----------
        device : str
            The device to use for the simulation.
        Returns
        -------
        inputs : (batch, T, 2), [heading, speed] at each time step
        positions : (batch, T, 2), ground-truth (x,y) positions
        place_cell_activations : (batch, T, num_place_cells), ground-truth place cell activations
        """
        # --- initial position & velocity ----------------------------------------
        pos = torch.rand(self.num_trajectories, 2, device=device) * self.arena_size
        # sample initial heading uniformly in (0, 2pi), speed around mu_speed
        hd0 = torch.rand(self.num_trajectories, device=device) * 2 * torch.pi
        spd0 = torch.clamp(
            torch.randn(self.num_trajectories, device=device) * self.sigma_speed
            + self.mu_speed,
            min=0.0,
        )
        vel = torch.stack((torch.cos(hd0), torch.sin(hd0)), dim=-1) * spd0.unsqueeze(-1)

        pos_all, vel_all = [pos], [vel]

        sqrt_2dt_over_tau = math.sqrt(2 * self.dt / self.tau_vel)
        for _ in range(self.num_time_steps - 1):
            # OU velocity update (momentum)
            noise = torch.randn_like(vel)
            vel = (
                vel
                + (self.dt / self.tau_vel) * (-vel)
                + self.sigma_speed * sqrt_2dt_over_tau * noise
            )

            # position update
            pos = pos + vel * self.dt

            # --- reflective boundaries -----------------------------------------
            out_left = pos[:, 0] < 0
            out_right = pos[:, 0] > self.arena_size
            out_bottom = pos[:, 1] < 0
            out_top = pos[:, 1] > self.arena_size

            # reflect positions and flip corresponding velocity component
            if out_left.any():
                pos[out_left, 0] *= -1
                vel[out_left, 0] *= -1
            if out_right.any():
                pos[out_right, 0] = 2 * self.arena_size - pos[out_right, 0]
                vel[out_right, 0] *= -1
            if out_bottom.any():
                pos[out_bottom, 1] *= -1
                vel[out_bottom, 1] *= -1
            if out_top.any():
                pos[out_top, 1] = 2 * self.arena_size - pos[out_top, 1]
                vel[out_top, 1] *= -1

            pos_all.append(pos)
            vel_all.append(vel)

        vel_all = torch.stack(vel_all, 1)  # (batch, T, 2)
        pos_all = torch.stack(pos_all, 1)  # (batch, T, 2)
        speeds = torch.linalg.norm(vel_all, dim=-1)
        headings = torch.atan2(vel_all[..., 1], vel_all[..., 0]) % (2 * torch.pi)

        inputs = torch.stack((headings, speeds), dim=-1)  # (batch, T, 2)
        
        # Compute place cell activations as targets
        place_cell_activations = self.get_place_cell_activations(pos_all)
        
        return inputs, pos_all, place_cell_activations

    def setup(self, stage=None) -> None:
        inputs, positions, place_cell_activations = self.simulate_trajectories(device="cpu")
        full_dataset = TensorDataset(inputs, positions, place_cell_activations)

        # split into train and val
        train_size = int(self.train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )
