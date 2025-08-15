"""Hypergraph RNN."""

import lightning as L
import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np


class Hypergraph:
    """Represents a hypergraph with nodes and hyperedges."""

    def __init__(
        self,
        *,
        num_nodes: int,
        hyperedges: List[List[int]],
    ) -> None:
        """
        Initialize hypergraph.

        :param num_nodes: Number of nodes in the hypergraph
        :param hyperedges: List of hyperedges, each is a list of node indices
                          Example: [[0, 1], [1, 2, 3], [0, 3]] for 3 hyperedges
        """
        self.num_nodes = num_nodes
        self.hyperedges = hyperedges
        self.num_hyperedges = len(hyperedges)

        # Create incidence matrix I: (num_nodes, num_hyperedges)
        self.incidence_matrix = self._build_incidence_matrix()

    def _build_incidence_matrix(self) -> torch.Tensor:
        """Build the incidence matrix I mapping nodes to hyperedges."""
        incidence_matrix = torch.zeros(
            self.num_nodes, self.num_hyperedges, dtype=torch.float32
        )

        for he_idx, hyperedge in enumerate(self.hyperedges):
            for node_idx in hyperedge:
                incidence_matrix[node_idx, he_idx] = 1.0

        return incidence_matrix

    @classmethod
    def create_random_hypergraph(
        cls,
        num_nodes: int,
        num_hyperedges: int,
        max_hyperedge_size: int = 3,
    ) -> "Hypergraph":
        """Create a random hypergraph for testing."""
        hyperedges = []

        for _ in range(num_hyperedges):
            # Random hyperedge size (2 or 3 nodes)
            size = np.random.randint(2, max_hyperedge_size + 1)
            # Random nodes for this hyperedge
            nodes = np.random.choice(num_nodes, size, replace=False).tolist()
            hyperedges.append(nodes)

        return cls(num_nodes=num_nodes, hyperedges=hyperedges)

    def to_device(self, device):
        """Move incidence matrix to device."""
        self.incidence_matrix = self.incidence_matrix.to(device)
        return self


class HypergraphRNNStep(nn.Module):
    """HypergraphRNN step with nodes and hyperedges."""

    def __init__(
        self,
        *,
        hypergraph: Hypergraph,
        input_dim: int,
        alpha_node: float,
        alpha_hyperedge: float,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        """Initialize the HypergraphRNN step.

        :param hypergraph: The hypergraph structure
        :param input_dim: Dimension of the input signal
        :param alpha_node: Node update rate
        :param alpha_hyperedge: Hyperedge update rate
        :param activation: Activation function
        """
        super().__init__()

        self.hypergraph = hypergraph
        self.input_dim = input_dim
        self.alpha_node = alpha_node
        self.alpha_hyperedge = alpha_hyperedge
        self.activation = activation()

        # Node parameters
        self.W_node_rec = nn.Linear(
            hypergraph.num_nodes, hypergraph.num_nodes
        )  # Node recurrent weights
        self.W_node_in = nn.Linear(
            input_dim, hypergraph.num_nodes
        )  # Node input weights

        # Hyperedge parameters
        self.W_hyperedge_rec = nn.Linear(
            hypergraph.num_hyperedges, hypergraph.num_hyperedges
        )  # Hyperedge recurrent weights
        self.W_hyperedge_in = nn.Linear(
            input_dim, hypergraph.num_hyperedges
        )  # Hyperedge input weights

        self.hyperedge_to_node_weight = nn.Parameter(torch.ones(1))
        self.node_to_hyperedge_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        *,
        node_states: torch.Tensor,  # (batch_size, num_nodes)
        hyperedge_states: torch.Tensor,  # (batch_size, num_hyperedges)
        inputs: torch.Tensor,  # (batch_size, input_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the HypergraphRNN step."""

        incidence_matrix = self.hypergraph.incidence_matrix.to(
            node_states.device
        )  # (num_nodes, num_hyperedges)

        node_recurrent = self.W_node_rec(node_states)  # (batch, num_nodes)
        node_input = self.W_node_in(inputs)  # (batch, num_nodes)

        # Hyperedge → Node messages
        hyperedge_to_node = self.hyperedge_to_node_weight * torch.matmul(
            hyperedge_states, incidence_matrix.T
        )

        new_node_states = (
            1 - self.alpha_node
        ) * node_states + self.alpha_node * self.activation(
            node_recurrent + hyperedge_to_node + node_input
        )

        hyperedge_recurrent = self.W_hyperedge_rec(
            hyperedge_states
        )  # (batch, num_hyperedges)
        hyperedge_input = self.W_hyperedge_in(inputs)  # (batch, num_hyperedges)

        # Node → Hyperedge messages
        node_to_hyperedge = self.node_to_hyperedge_weight * torch.matmul(
            node_states, incidence_matrix
        )  # (batch, num_hyperedges)

        new_hyperedge_states = (
            1 - self.alpha_hyperedge
        ) * hyperedge_states + self.alpha_hyperedge * self.activation(
            hyperedge_recurrent + node_to_hyperedge + hyperedge_input
        )

        return new_node_states, new_hyperedge_states


class HypergraphRNN(nn.Module):
    """Hypergraph RNN model with encoder, decoder, and readout."""

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        hypergraph: Hypergraph,
        alpha_node: float,
        alpha_hyperedge: float,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        """Initialize the Hypergraph RNN model.

        :param input_size: Dimension of the input signal
        :param output_size: Dimension of the output
        :param hypergraph: The hypergraph structure defining connectivity
        :param alpha_node: Node update rate
        :param alpha_hyperedge: Hyperedge update rate
        :param activation: Activation function for RNN step
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Hypergraph RNN step (equivalent to RNNStep)
        self.hypergraph_step = HypergraphRNNStep(
            hypergraph=hypergraph,
            input_dim=input_size,
            alpha_node=alpha_node,
            alpha_hyperedge=alpha_hyperedge,
            activation=activation,
        )

        # Output layer (equivalent to W_out)
        self.W_out = nn.Linear(hypergraph.num_nodes, output_size)

        # Layer to initialize node and hyperedge states (equivalent to W_h_init)
        self.W_node_init = nn.Linear(2, hypergraph.num_nodes)
        self.W_hyperedge_init = nn.Linear(2, hypergraph.num_hyperedges)

        self.initialize_weights()

    def forward(
        self, inputs: torch.Tensor, pos_0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Hypergraph RNN.

        :param inputs: Input sequence (batch_size, time_steps, input_size)
        :param pos_0: Initial position (batch_size, 2)
        :return: (hidden_states, outputs)
        """
        # inputs has shape (batch_size, time_steps, input_size)
        # pos_0 has shape (batch_size, 2)

        hidden_states = []  # Will store node states over time
        outputs = []

        # Initialize node and hyperedge states from pos_0
        node_states = torch.tanh(self.W_node_init(pos_0))  # (batch_size, num_nodes)
        hyperedge_states = torch.tanh(
            self.W_hyperedge_init(pos_0)
        )  # (batch_size, num_hyperedges)

        # Loop over time steps (same structure as vanilla RNN)
        for t in range(inputs.shape[1]):
            input_t = inputs[:, t, :]  # (batch_size, input_size)

            # Update states using hypergraph step
            node_states, hyperedge_states = self.hypergraph_step(
                node_states=node_states,
                hyperedge_states=hyperedge_states,
                inputs=input_t,
            )

            # Store node states and compute output
            hidden_states.append(node_states)
            outputs.append(self.W_out(node_states))

        return torch.stack(hidden_states, dim=1), torch.stack(outputs, dim=1)

    def initialize_weights(self) -> None:
        """Initialize weights for stable training (similar to vanilla RNN)"""
        # Output weights
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)

        # Initial state encoders
        nn.init.xavier_uniform_(self.W_node_init.weight)
        nn.init.zeros_(self.W_node_init.bias)
        nn.init.xavier_uniform_(self.W_hyperedge_init.weight)
        nn.init.zeros_(self.W_hyperedge_init.bias)


class HypergraphRNNLightning(L.LightningModule):
    """Lightning module for the Hypergraph RNN models."""

    def __init__(
        self,
        model: HypergraphRNN,
        learning_rate: float,
        step_size: int,
        gamma: float,
        weight_decay: float,
    ) -> None:
        """Initialize the HypergraphRNNLightning model."""
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.weight_decay = weight_decay

    def training_step(self, batch) -> torch.Tensor:
        inputs, targets = batch
        # inputs has shape (batch_size, time_steps, input_size)
        # targets has shape (batch_size, time_steps, output_size)
        hidden_states, outputs = self.model(inputs=inputs, pos_0=targets[:, 0, :])

        loss = nn.functional.mse_loss(outputs, targets)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch) -> torch.Tensor:
        inputs, targets = batch
        hidden_states, outputs = self.model(inputs=inputs, pos_0=targets[:, 0, :])

        loss = nn.functional.mse_loss(outputs, targets)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        """Configure the optimizer and scheduler for the Hypergraph RNN model."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }
