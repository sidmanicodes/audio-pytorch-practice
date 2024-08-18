# Imports
import os
import torch
from torch import nn
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from typing import Tuple
import matplotlib.pyplot as plt
import torchaudio
from cnn import CNN
from urban_sound_dataset import UrbanSoundDataset

# Constants
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
ANNOTATIONS_FILE_PATH = os.path.join("data", "metadata", "UrbanSound8K.csv")
AUDIO_FILE_PATH = os.path.join("data", "audio")
STANDARD_SR = 22_050
STANDARD_NUM_SAMPLES = 22_050
FRAME_SIZE = 1_024
HOP_SIZE = 512
NUM_MEL_BANDS = 62

history = []

def train_one_epoch(
        model: CNN, 
        data_loader: torch.utils.data.DataLoader, 
        loss_fn: nn.modules.loss._Loss, 
        optimizer: torch.optim.Optimizer, 
        device: str
        ) -> None:
    for inputs, labels in data_loader:
        # Place tensors on appropriate hardware
        inputs, labels = inputs.to(device), labels.to(device)

        # Calculate loss
        logits = model(inputs)
        loss = loss_fn(logits, labels)

        # Backpropagate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    history.append(loss.item())
    print(f"Loss: {loss.item()}")
    
def train(
        model: CNN, 
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
    # Get device
    device = "cpu"

    if torch.backends.mps.is_available():
        device = "mps"

    print(f"Using {device}")

    # Create Mel Spectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=STANDARD_SR,
        n_fft=FRAME_SIZE,
        hop_length=HOP_SIZE,
        n_mels=NUM_MEL_BANDS,
    )

    # Load dataset
    usd = UrbanSoundDataset(
        annotations_file_path=ANNOTATIONS_FILE_PATH, 
        audio_file_path=AUDIO_FILE_PATH,
        device=device,
        standard_sr=STANDARD_SR,
        standard_num_samples=STANDARD_NUM_SAMPLES,
        transformation=mel_spectrogram
    )

    if usd:
        print("Successfully downloaded training data!")
    else:
        raise Exception("Something went wrong when downloading the data")
    
    # Create the data loader
    data_loader = DataLoader(
        dataset=usd,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # num_workers=4
    )

    # Build the model
    cnn = CNN().to(device)

    # Instantiate model loss and optimizer
    cross_entropy_loss = nn.CrossEntropyLoss()
    adam_optim = torch.optim.Adam(params=cnn.parameters(), lr=LEARNING_RATE)

    # Train the model
    train(
        model=cnn, 
        data_loader=data_loader, 
        loss_fn=cross_entropy_loss, 
        optimizer=adam_optim, 
        device=device, 
        epochs=NUM_EPOCHS
    )

    # Save model
    torch.save(cnn.state_dict(), "urban_sound_classifier.pth")

    # Visualize loss curve by epoch (comment out to graph)
    plt.figure(figsize=(25, 10))
    plt.plot(range(1, NUM_EPOCHS+1), history, color='r')
    plt.show()