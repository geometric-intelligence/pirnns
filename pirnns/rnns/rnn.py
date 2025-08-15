import torch
import torch.nn as nn
import lightning as L


class RNNStep(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        alpha: float,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        """
        A single time step of the RNN.
        """
        super(RNNStep, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.activation = activation()

        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        h = (1 - self.alpha) * hidden + self.alpha * self.activation(
            self.W_in(input) + self.W_rec(hidden)
        )
        return h


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        alpha: float = 0.1,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        """
        Initialize the Path Integrating RNN.
        :param input_size: The size of the velocity input (= dimension of space).
        :param hidden_size: The size of the hidden state (number of neurons/"grid cells").
        :param output_size: The size of the output vector (dimension of space).
        :param alpha: RNN update rate.
        :param activation: The activation function.
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn_step = RNNStep(input_size, hidden_size, alpha, activation)
        self.W_out = nn.Linear(hidden_size, output_size)

        # Layer to initialize hidden state
        self.W_h_init = nn.Linear(2, hidden_size)

        self.initialize_weights()

    def forward(
        self, inputs: torch.Tensor, pos_0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # inputs has shape (batch_size, time_steps, input_size)
        # pos_0 has shape (batch_size, 2)
        hidden_states = []
        outputs = []
        hidden = torch.tanh(self.W_h_init(pos_0))
        for t in range(inputs.shape[1]):
            input_t = inputs[:, t, :]
            hidden = self.rnn_step(input_t, hidden)
            hidden_states.append(hidden)
            outputs.append(self.W_out(hidden))
        return torch.stack(hidden_states, dim=1), torch.stack(outputs, dim=1)

    def initialize_weights(self) -> None:
        """Initialize weights for stable RNN training"""
        # 1. Input weights (W_in) - Xavier initialization
        nn.init.xavier_uniform_(self.rnn_step.W_in.weight)
        nn.init.zeros_(self.rnn_step.W_in.bias)

        # 2. Recurrent weights (W_rec) - Orthogonal initialization
        nn.init.orthogonal_(self.rnn_step.W_rec.weight)
        nn.init.zeros_(self.rnn_step.W_rec.bias)

        # 3. Output weights (W_out) - Xavier initialization
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)

        # 4. Initial hidden state encoder (W_h_init) - Xavier initialization
        nn.init.xavier_uniform_(self.W_h_init.weight)
        nn.init.zeros_(self.W_h_init.bias)


class RNNLightning(L.LightningModule):
    def __init__(
        self,
        model: RNN,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        step_size: int = 100,
        gamma: float = 0.5,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma

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
