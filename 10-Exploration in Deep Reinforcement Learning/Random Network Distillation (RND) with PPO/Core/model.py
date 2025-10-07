# model.py
from abc import ABC
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


def conv_shape(input_size, kernel_size, stride, padding=0):
    return (input_size + 2 * padding - kernel_size) // stride + 1


# === Policy Network (unchanged) ===
class PolicyModel(nn.Module, ABC):
    def __init__(self, state_shape, n_actions):
        super().__init__()
        c, w, h = state_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        flatten_dim = 32 * conv_shape(w, 3, 1, 1) * conv_shape(h, 3, 1, 1)

        self.fc1 = nn.Linear(flatten_dim, 256)
        self.gru = nn.GRUCell(256, 256)

        self.value_extras = nn.Linear(256, 256)
        self.policy_extras = nn.Linear(256, 256)

        self.policy_head = nn.Linear(256, n_actions)
        self.int_value_head = nn.Linear(256, 1)
        self.ext_value_head = nn.Linear(256, 1)

        # Orthogonal init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, obs, hidden):
        # obs: (B, C, W, H) or (B, 1, C, W, H)
        if obs.ndim == 5:
            obs = obs.squeeze(1)

        x = obs.float() / 255.0
        x = self.conv(x)
        x = F.relu(self.fc1(x))
        h = self.gru(x, hidden)

        # separate pathways for value vs policy
        v = h + F.relu(self.value_extras(h))
        p = h + F.relu(self.policy_extras(h))

        int_v = self.int_value_head(v)
        ext_v = self.ext_value_head(v)

        logits = self.policy_head(p)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        return dist, int_v, ext_v, probs, h


# === Target Model ===
class TargetModel(nn.Module, ABC):
    def __init__(self, state_shape):
        super().__init__()
        c, w, h = state_shape

        # shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1),  # → 32×⌈w/2⌉×⌈h/2⌉
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # → 64×⌈w/4⌉×⌈h/4⌉
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # → 64×⌈w/8⌉×⌈h/8⌉
            nn.ReLU(),
            nn.Flatten()
        )

        # project to 512-dim feature
        # after three strides of 2 on 7×7 → 1×1, so flatten dim = 64
        self.project = nn.Linear(64, 512)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                m.bias.data.zero_()

    def forward(self, obs):
        x = obs.float() / 255.0
        x = self.encoder(x)
        return self.project(x)


# === Predictor Model ===
class PredictorModel(nn.Module, ABC):
    def __init__(self, state_shape):
        super().__init__()
        c, w, h = state_shape

        # same encoder as TargetModel
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # extra MLP
        self.fc_hidden = nn.Linear(64, 512)
        self.fc_out = nn.Linear(512, 512)
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # slow down final layer a bit:
                gain = np.sqrt(0.01) if m is self.fc_out else np.sqrt(2)
                nn.init.orthogonal_(m.weight, gain=gain)
                m.bias.data.zero_()

    def forward(self, obs):
        x = obs.float() / 255.0
        x = self.encoder(x)
        x = self.relu(self.fc_hidden(x))
        return self.fc_out(x)
