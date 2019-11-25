import torch
from torch import nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, movement_data_len=0):
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, padding=2, stride=5)
        self.conv2 = nn.Conv2d(20, 50, 5, padding=2, stride=5)
        self.conv3 = nn.Conv2d(50, 1000, 5, padding=2, stride=5)
        self.conv4 = nn.Conv2d(1000, 2000, 3)

        self.movement_data_len = movement_data_len

        self.unconv1 = nn.ConvTranspose2d(
            2000 + movement_data_len, 1000 + movement_data_len // 2, 3
        )

        self.unconv2 = nn.ConvTranspose2d(
            1000 + movement_data_len // 2,
            50 + movement_data_len // 4,
            5,
            padding=2,
            stride=5,
        )
        self.unconv3 = nn.ConvTranspose2d(
            50 + movement_data_len // 4,
            20 + movement_data_len // 8,
            5,
            padding=1,
            stride=5,
        )
        self.unconv4 = nn.ConvTranspose2d(
            20 + movement_data_len // 8, 3, 4, padding=4, stride=5
        )

        self.encoding = None

    def is_same_at_start(self, other):
        is_eq = isinstance(other, AutoEncoder)
        is_eq = self.movement_data_len == other.movement_data_len and is_eq
        return is_eq

    def set_movement_data_len(self, movement_data_len):
        if movement_data_len != self.movement_data_len:
            self.movement_data_len = movement_data_len

            self.unconv1 = nn.ConvTranspose2d(
                200 + movement_data_len, 100 + movement_data_len // 2, 3
            )
            self.unconv2 = nn.ConvTranspose2d(
                100 + movement_data_len // 2,
                50 + movement_data_len // 4,
                5,
                padding=2,
                stride=5,
            )
            self.unconv3 = nn.ConvTranspose2d(
                50 + movement_data_len // 4,
                20 + movement_data_len // 8,
                5,
                padding=1,
                stride=5,
            )
            self.unconv4 = nn.ConvTranspose2d(
                20 + movement_data_len // 8, 3, 4, padding=4, stride=5
            )

    def forward(self, x, movement):
        x = self.conv1.forward(x)
        x = self.conv2.forward(x)
        x = self.conv3.forward(x)
        x = self.conv4.forward(x)

        x = torch.cat((x, movement), dim=1)
        self.encoding = x.clone()

        x = self.unconv1.forward(x)
        x = self.unconv2.forward(x)
        x = self.unconv3.forward(x)
        x = self.unconv4.forward(x)

        return x
