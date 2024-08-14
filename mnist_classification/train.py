# Imports
import torch
from torch import nn
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from typing import Tuple
import matplotlib.pyplot as plt

# Constants
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

history = []

class FeedForwardNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_data: torch.tensor) -> torch.tensor:
        """
        Returns predictions from a forward pass through the network

        Arguments
        ---------
        - input_data | torch.tensor [shape=(1, 28, 28)]
            - Input data (images from MNIST)

        Returns
        -------
        - predictions | torch.tensor [shape=(10,)]
            - A tensor of probabilities of the image being a number ranging from 0-9
        """
        # Flatten images
        flattened_data = self.flatten(input_data)

        # Get the logits from forward pass
        logits = self.dense_layers(flattened_data)

        # Use softmax to convert the logits to probabilities
        predictions = self.softmax(logits)

        return predictions

def download_mnist_data() -> Tuple[MNIST, MNIST]:
    """
    Returns training and testing data partitions for the MNIST
    dataset
    """
    train_data = MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor()
    )

    test_data = MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )

    return train_data, test_data

def train_one_epoch(
        model: FeedForwardNet, 
        data_loader: torch.utils.data.DataLoader, 
        loss_fn: nn.modules.loss._Loss, 
        optimizer: torch.optim.Optimizer, 
        device: str
        ) -> None:
    for inputs, labels in data_loader:
        # Place tensors on appropriate hardware
        inputs, labels = inputs.to(device), labels.to(device)

        # Calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, labels)

        # Backpropagate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    history.append(loss.item())
    print(f"Loss: {loss.item()}")
    
def train(
        model: FeedForwardNet, 
        data_loader: torch.utils.data.DataLoader, 
        loss_fn: nn.modules.loss._Loss, 
        optimizer: torch.optim.Optimizer, 
        device: str,
        epochs: int
        ) -> None:
    # Train the model for the number of epochs
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model=model, data_loader=data_loader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        print("-------------")
    print("Training complete!")

if __name__ == "__main__":
    # Download data
    training_data, _ = download_mnist_data()

    if training_data:
        print("Successfully downloaded training data!")
    else:
        raise Exception("Something went wrong when downloading the data")
    
    # Create the data loader
    data_loader = DataLoader(
        dataset=training_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    # Build the model
    device = "cpu"

    if torch.backends.mps.is_available():
        device = "mps"

    print(f"Using {device}")

    feed_forward_net = FeedForwardNet().to(device)

    # Instantiate model loss and optimizer
    cross_entropy_loss = nn.CrossEntropyLoss()
    adam_optim = torch.optim.Adam(params=feed_forward_net.parameters(), lr=LEARNING_RATE)

    # Train the model
    train(
        model=feed_forward_net, 
        data_loader=data_loader, 
        loss_fn=cross_entropy_loss, 
        optimizer=adam_optim, 
        device=device, 
        epochs=NUM_EPOCHS
    )

    # Save model
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")

    # Visualize loss curve by epoch (comment out to graph)
    # plt.figure(figsize=(25, 10))
    # plt.plot(range(1, NUM_EPOCHS+1), history, color='r')
    # plt.show()