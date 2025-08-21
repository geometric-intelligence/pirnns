import torch
import torch.nn as nn
import lightning as L


class CoupledRNNStep(nn.Module):
    """
    Coupled RNN step with two interacting subpopulations.

    Population 1: e.g., "neurons"
    Population 2: e.g., "astrocytes"
    """

    def __init__(
        self,
        input_size: int,
        pop1_size: int,
        pop2_size: int,
        alpha1: float,
        alpha2: float,
        activation: type[nn.Module] = nn.Tanh,
    ):
        """
        Initialize the Coupled RNN step.
        :param input_size: The size of the velocity input (= dimension of space).
        :param pop1_size: The size of the first population (e.g., neurons).
        :param pop2_size: The size of the second population (e.g., astrocytes).
        :param alpha1: RNN update rate for the first population.
        :param alpha2: RNN update rate for the second population.
        :param activation: The activation function.
        """
        super().__init__()

        self.pop1_size = pop1_size
        self.pop2_size = pop2_size
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.activation = activation()

        # Input connections
        self.W_in_pop1 = nn.Linear(input_size, pop1_size)
        self.W_in_pop2 = nn.Linear(input_size, pop2_size)

        # Within-population recurrent connections
        self.W_rec_pop1 = nn.Linear(pop1_size, pop1_size)
        self.W_rec_pop2 = nn.Linear(pop2_size, pop2_size)

        # Cross-population connections
        self.W_pop1_to_pop2 = nn.Linear(pop1_size, pop2_size)
        self.W_pop2_to_pop1 = nn.Linear(pop2_size, pop1_size)

    def forward(
        self,
        input: torch.Tensor,
        hidden1: torch.Tensor,
        hidden2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both populations.

        :param input: (batch, input_size)
        :param hidden1: (batch, pop1_size)
        :param hidden2: (batch, pop2_size)

        :return: new_hidden1: (batch, pop1_size)
        :return: new_hidden2: (batch, pop2_size)
        """

        # Compute updates for population 1
        input_contrib1 = self.W_in_pop1(input)
        recurrent_contrib1 = self.W_rec_pop1(hidden1)
        cross_contrib1 = self.W_pop2_to_pop1(hidden2)  # Input from pop2

        total_input1 = input_contrib1 + recurrent_contrib1 + cross_contrib1
        new_hidden1 = (1 - self.alpha1) * hidden1 + self.alpha1 * self.activation(
            total_input1
        )

        # Compute updates for population 2
        input_contrib2 = self.W_in_pop2(input)
        recurrent_contrib2 = self.W_rec_pop2(hidden2)
        cross_contrib2 = self.W_pop1_to_pop2(hidden1)  # Input from pop1

        total_input2 = input_contrib2 + recurrent_contrib2 + cross_contrib2
        new_hidden2 = (1 - self.alpha2) * hidden2 + self.alpha2 * self.activation(
            total_input2
        )

        return new_hidden1, new_hidden2


class CoupledRNN(nn.Module):
    """
    Coupled RNN with two interacting populations.
    """

    def __init__(
        self,
        input_size: int,
        pop1_size: int,
        pop2_size: int,
        output_size: int,
        alpha1: float,
        alpha2: float,
        activation: type[nn.Module] = nn.Tanh,
        output_from: str = "both",  # "pop1", "pop2", or "both"
    ):
        """
        Initialize the Coupled RNN.
        :param input_size: The size of the velocity input (= dimension of space).
        :param pop1_size: The size of the first population (e.g., neurons).
        :param pop2_size: The size of the second population (e.g., astrocytes).
        :param output_size: The size of the output vector (number of place cells).
        :param alpha1: RNN update rate for the first population.
        :param alpha2: RNN update rate for the second population.
        :param activation: The activation function.
        :param output_from: The population to use for the output.
        """
        super().__init__()

        self.pop1_size = pop1_size
        self.pop2_size = pop2_size
        self.output_size = output_size
        self.output_from = output_from

        self.rnn_step = CoupledRNNStep(
            input_size,
            pop1_size,
            pop2_size,
            alpha1,
            alpha2,
            activation,
        )

        if output_from == "pop1":
            self.W_out = nn.Linear(pop1_size, output_size, bias=False)
        elif output_from == "pop2":
            self.W_out = nn.Linear(pop2_size, output_size, bias=False)
        elif output_from == "both":
            self.W_out = nn.Linear(pop1_size + pop2_size, output_size, bias=False)
        else:
            raise ValueError("output_from must be 'pop1', 'pop2', or 'both'")

        # Initialization layers
        self.W_h1_init = nn.Linear(output_size, pop1_size, bias=False)
        self.W_h2_init = nn.Linear(output_size, pop2_size, bias=False)

    def forward(
        self, inputs: torch.Tensor, place_cells_0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the coupled RNN.

        :param inputs: (batch, time, input_size)
        :param place_cells_0: (batch, output_size) - initial state

        :return: hidden1_states: (batch, time, pop1_size)
        :return: hidden2_states: (batch, time, pop2_size)
        :return: outputs: (batch, time, output_size)
        """
        _, seq_len, _ = inputs.shape

        # Initialize hidden states
        hidden1 = self.W_h1_init(place_cells_0)
        hidden2 = self.W_h2_init(place_cells_0)

        # Store states
        hidden1_states = []
        hidden2_states = []
        outputs = []

        for t in range(seq_len):
            input_t = inputs[:, t, :]
            hidden1, hidden2 = self.rnn_step(input_t, hidden1, hidden2)

            # Compute output
            if self.output_from == "pop1":
                output_t = self.W_out(hidden1)
            elif self.output_from == "pop2":
                output_t = self.W_out(hidden2)
            elif self.output_from == "both":
                combined_hidden = torch.cat([hidden1, hidden2], dim=-1)
                output_t = self.W_out(combined_hidden)

            hidden1_states.append(hidden1)
            hidden2_states.append(hidden2)
            outputs.append(output_t)

        return (
            torch.stack(hidden1_states, dim=1),
            torch.stack(hidden2_states, dim=1),
            torch.stack(outputs, dim=1),
        )


class CoupledRNNLightning(L.LightningModule):
    def __init__(
        self,
        model: CoupledRNN,
        learning_rate: float,
        weight_decay: float,
        step_size: int,
        gamma: float,
    ) -> None:
        """
        Initialize the Coupled RNN Lightning module.
        :param model: The CoupledRNN model.
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
        hidden1_states, hidden2_states, outputs = self.model(
            inputs=inputs, place_cells_0=target_place_cells[:, 0, :]
        )

        # Cross-entropy loss
        y = target_place_cells.reshape(-1, self.model.output_size)
        yhat = torch.softmax(outputs.reshape(-1, self.model.output_size), dim=-1)
        loss = -(y * torch.log(yhat + 1e-8)).sum(-1).mean()

        # Weight regularization on all recurrent weights
        recurrent_weights_loss = (
            (self.model.rnn_step.W_rec_pop1.weight**2).sum()
            + (self.model.rnn_step.W_rec_pop2.weight**2).sum()
            + (self.model.rnn_step.W_pop1_to_pop2.weight**2).sum()
            + (self.model.rnn_step.W_pop2_to_pop1.weight**2).sum()
        )
        loss += self.weight_decay * recurrent_weights_loss

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
        hidden1_states, hidden2_states, outputs = self.model(
            inputs=inputs, place_cells_0=target_place_cells[:, 0, :]
        )

        # Cross-entropy loss
        y = target_place_cells.reshape(-1, self.model.output_size)
        yhat = torch.softmax(outputs.reshape(-1, self.model.output_size), dim=-1)
        loss = -(y * torch.log(yhat + 1e-8)).sum(-1).mean()

        # Weight regularization on all recurrent weights
        recurrent_weights_loss = (
            (self.model.rnn_step.W_rec_pop1.weight**2).sum()
            + (self.model.rnn_step.W_rec_pop2.weight**2).sum()
            + (self.model.rnn_step.W_pop1_to_pop2.weight**2).sum()
            + (self.model.rnn_step.W_pop2_to_pop1.weight**2).sum()
        )
        loss += self.weight_decay * recurrent_weights_loss

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
        """Configure the optimizer and scheduler for the Coupled RNN model."""
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
