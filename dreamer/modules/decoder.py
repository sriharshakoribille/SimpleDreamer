import torch.nn as nn
from dreamer.utils.utils import build_network
from dreamer.utils.utils import (
    initialize_weights,
    horizontal_forward,
    create_normal_dist,
)


class Decoder(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer.decoder
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        activation = getattr(nn, self.config.activation)()
        self.observation_shape = observation_shape

        self.network = nn.Sequential(
            nn.Linear(
                self.deterministic_size + self.stochastic_size, self.config.depth * 32
            ),
            nn.Unflatten(1, (self.config.depth * 32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(
                self.config.depth * 32,
                self.config.depth * 4,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.ConvTranspose2d(
                self.config.depth * 4,
                self.config.depth * 2,
                self.config.kernel_size,
                self.config.stride,
            ),
            activation,
            nn.ConvTranspose2d(
                self.config.depth * 2,
                self.config.depth * 1,
                self.config.kernel_size + 1,
                self.config.stride,
            ),
            activation,
            nn.ConvTranspose2d(
                self.config.depth * 1,
                self.observation_shape[0],
                self.config.kernel_size + 1,
                self.config.stride,
            ),
        )
        self.network.apply(initialize_weights)

    def forward(self, posterior, deterministic):
        x = horizontal_forward(
            self.network, posterior, deterministic, output_shape=self.observation_shape
        )
        dist = create_normal_dist(x, std=1, event_shape=len(self.observation_shape))
        return dist

class Decoder_state(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer.decoder
        self.observation_shape = observation_shape
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.network = build_network(
            self.stochastic_size+self.deterministic_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            self.observation_shape[0],
        )

    def forward(self, posterior, deterministic):
        x = horizontal_forward(
            self.network, posterior, deterministic, output_shape=self.observation_shape
        )
        dist = create_normal_dist(x, std=1, event_shape=len(self.observation_shape))
        return dist
