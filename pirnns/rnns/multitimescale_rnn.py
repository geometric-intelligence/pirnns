import torch
import torch.nn as nn
import lightning as L


class MultiTimescaleRNNStep(nn.Module):
    """
    Multi-timescale RNN step where each hidden unit has its own update rate (timescale).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dt: float,
        timescales: torch.Tensor,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        """
        Initialize the Multi-timescale RNN step.

        :param input_size: The size of the velocity input (= dimension of space).
        :param hidden_size: The size of the hidden state (number of neurons).
        :param dt: The time step.
        :param timescales: Tensor of shape (hidden_size,) containing timescales for each unit.
        :param activation: The activation function.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt
        self.activation = activation()

        # Register timescales and alphas as buffers (not trainable parameters)
        self.register_buffer("timescales", timescales)
        alphas = 1 - torch.exp(-dt / timescales)
        self.register_buffer("alphas", alphas)

        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with per-unit update rates.

        :param input: (batch, input_size)
        :param hidden: (batch, hidden_size)
        :return: new_hidden: (batch, hidden_size)
        """
        pre_activation = self.W_in(input) + self.W_rec(hidden)
        activated = self.activation(pre_activation)

        new_hidden = (1 - self.alphas) * hidden + self.alphas * activated

        return new_hidden


class MultiTimescaleRNN(nn.Module):
    """
    Multi-timescale RNN where each hidden unit has its own update rate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dt: float,
        timescales_config: dict,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        """
        Initialize the Multi-timescale RNN.

        :param input_size: The size of the velocity input (= dimension of space).
        :param hidden_size: The size of the hidden state (number of neurons).
        :param output_size: The size of the output vector (number of place cells).
        :param dt: The time step size.
        :param timescales_config: Dictionary specifying how to set the timescales.
        :param activation: The activation function.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dt = dt

        # Generate timescales based on configuration
        timescales = self._generate_timescales(hidden_size, timescales_config)

        self.rnn_step = MultiTimescaleRNNStep(
            input_size, hidden_size, dt, timescales, activation
        )
        self.W_out = nn.Linear(hidden_size, output_size, bias=False)

        # Layer to initialize hidden state
        self.W_h_init = nn.Linear(output_size, hidden_size, bias=False)

        self._initialize_weights()

    def _generate_timescales(
        self, hidden_size: int, timescales_config: dict
    ) -> torch.Tensor:
        """
        Generate timescales based on configuration.

        :param hidden_size: Number of hidden units
        :param timescales_config: Configuration dictionary
        :return: Tensor of timescales of shape (hidden_size,)
        """
        timescale_type = timescales_config["type"]

        if timescale_type == "discrete":
            discrete_values = timescales_config["values"]
            discrete_values = torch.tensor(discrete_values, dtype=torch.float32)

            if len(discrete_values) == 1:
                timescales = discrete_values.repeat(hidden_size)
            else:
                # Each timescale picked uniformly from K discrete values
                indices = torch.randint(0, len(discrete_values), (hidden_size,))
                timescales = discrete_values[indices]

        elif timescale_type == "continuous":
            distribution = timescales_config["distribution"]

            if distribution == "uniform":
                min_timescale = float(timescales_config["min_timescale"])
                max_timescale = float(timescales_config["max_timescale"])
                timescales = torch.uniform(
                    min_timescale, max_timescale, size=(hidden_size,)
                )

            elif distribution == "powerlaw":
                # Power-law distribution: P(x) ∝ x^(-α)
                # We use inverse transform sampling since PyTorch doesn't have a built-in power-law
                exponent = float(timescales_config["exponent"])
                min_timescale = float(timescales_config["min_timescale"])
                max_timescale = float(timescales_config["max_timescale"])

                # Generate uniform random samples
                u = torch.rand(hidden_size)

                # Inverse transform sampling for bounded power-law
                # F^(-1)(u) = x_min * [(x_max/x_min)^(1-α) * u + (1-u)]^(1/(1-α))
                if abs(exponent - 1.0) < 1e-6:
                    # Special case when α ≈ 1 (log-uniform distribution)
                    timescales = min_timescale * torch.pow(
                        torch.tensor(max_timescale / min_timescale), u
                    )
                else:
                    # General case
                    ratio = max_timescale / min_timescale
                    exponent_term = 1.0 - exponent
                    power_term = (ratio**exponent_term) * u + (1 - u)
                    timescales = min_timescale * torch.pow(
                        power_term, 1.0 / exponent_term
                    )

            else:
                raise ValueError(f"Unknown continuous distribution: {distribution}")

        return timescales

    def forward(
        self,
        inputs: torch.Tensor,
        place_cells_0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the multi-timescale RNN.

        :param inputs: (batch, time, input_size)
        :param place_cells_0: (batch, output_size) - used to initialize the hidden state

        :return: hidden_states: (batch, time, hidden_size)
        :return: outputs: (batch, time, output_size)
        """
        _, seq_len, _ = inputs.shape

        # Initialize hidden state
        hidden_states = []
        outputs = []
        hidden = self.W_h_init(place_cells_0)

        for t in range(seq_len):
            input_t = inputs[:, t, :]
            hidden = self.rnn_step(input_t, hidden)
            hidden_states.append(hidden)
            outputs.append(self.W_out(hidden))

        return torch.stack(hidden_states, dim=1), torch.stack(outputs, dim=1)

    def _initialize_weights(self) -> None:
        """Initialize weights for stable RNN training"""
        # 1. Input weights (W_in) - Xavier initialization
        nn.init.xavier_uniform_(self.rnn_step.W_in.weight)
        nn.init.zeros_(self.rnn_step.W_in.bias)

        # 2. Recurrent weights (W_rec) - Orthogonal initialization
        nn.init.orthogonal_(self.rnn_step.W_rec.weight)
        nn.init.zeros_(self.rnn_step.W_rec.bias)

        # 3. Output weights (W_out) - Xavier initialization
        nn.init.xavier_uniform_(self.W_out.weight)

        # 4. Initial hidden state encoder (W_h_init) - Xavier initialization
        nn.init.xavier_uniform_(self.W_h_init.weight)

    def get_timescale_stats(self) -> dict:
        """Return statistics about the timescale values."""
        timescales = self.rnn_step.timescales
        alphas = self.rnn_step.alphas
        return {
            "timescale_min": timescales.min().item(),
            "timescale_max": timescales.max().item(),
            "timescale_mean": timescales.mean().item(),
            "timescale_std": timescales.std().item(),
            "alpha_min": alphas.min().item(),
            "alpha_max": alphas.max().item(),
            "alpha_mean": alphas.mean().item(),
            "unique_timescales": len(torch.unique(timescales)),
        }


class MultiTimescaleRNNLightning(L.LightningModule):
    def __init__(
        self,
        model: MultiTimescaleRNN,
        learning_rate: float,
        weight_decay: float,
        step_size: int,
        gamma: float,
    ) -> None:
        """
        Initialize the Multi-timescale RNN Lightning module.

        :param model: The MultiTimescaleRNN model.
        :param learning_rate: The learning rate.
        :param weight_decay: The weight decay for the recurrent weights.
        :param step_size: The step size for the learning rate scheduler.
        :param gamma: The gamma for the learning rate scheduler.
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma

    def training_step(self, batch) -> torch.Tensor:
        inputs, target_positions, target_place_cells = batch
        hidden_states, outputs = self.model(
            inputs=inputs, place_cells_0=target_place_cells[:, 0, :]
        )

        # Cross-entropy loss
        y = target_place_cells.reshape(-1, self.model.output_size)
        yhat = torch.softmax(outputs.reshape(-1, self.model.output_size), dim=-1)
        loss = (
            -(y * torch.log(yhat + 1e-8)).sum(-1).mean()
        )  # Add small epsilon for numerical stability

        # Weight regularization on recurrent weights
        loss += self.weight_decay * (self.model.rnn_step.W_rec.weight**2).sum()

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
        inputs, target_positions, target_place_cells = batch
        hidden_states, outputs = self.model(
            inputs=inputs, place_cells_0=target_place_cells[:, 0, :]
        )

        # Cross-entropy loss
        y = target_place_cells.reshape(-1, self.model.output_size)
        yhat = torch.softmax(outputs.reshape(-1, self.model.output_size), dim=-1)
        loss = (
            -(y * torch.log(yhat + 1e-8)).sum(-1).mean()
        )  # Add small epsilon for numerical stability

        # Weight regularization on recurrent weights
        loss += self.weight_decay * (self.model.rnn_step.W_rec.weight**2).sum()

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
        """Configure the optimizer and scheduler for the Multi-timescale RNN model."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0,  # we do manual weight decay on recurrent weights in the loss
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

    def on_train_start(self):
        """Log timescale statistics at the start of training."""
        timescale_stats = self.model.get_timescale_stats()
        print(f"Timescale statistics: {timescale_stats}")
