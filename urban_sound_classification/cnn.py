from typing import Tuple
import torch
import torch.nn as nn
from torchinfo import summary

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()

        fc_in_features = self._get_conv_size(shape=(1, 64, 44))
        # print(f"Number of in-features to use: {fc_in_features}")
        self.fc = nn.Linear(in_features=fc_in_features, out_features=10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass the input through the convolutional layers
        x = self.conv_layers(x)

        # Flatten the input
        x = self.flatten(x)

        # Get the logits
        logits = self.fc(x)

        # Return the logits directly rather than passing them
        # through softmax (this is to prevent numeric instabilities)
        return logits

    def _get_conv_size(self, shape: Tuple[int]) -> int:
        with torch.no_grad():
            x = torch.randn(1, *shape)
            x = self.conv_layers(x)
            return x.numel()