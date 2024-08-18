import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

class UrbanSoundDataset(Dataset):
    def __init__(
            self, 
            annotations_file_path: str, 
            audio_file_path: str, 
            device: str,
            standard_sr: int, 
            standard_num_samples: int,
            transformation: torchaudio.transforms.MelSpectrogram
            ) -> None:
        self.annotations = pd.read_csv(annotations_file_path)
        self.audio_file_path = audio_file_path
        self.device = device
        self.standard_sr = standard_sr
        self.standard_num_samples = standard_num_samples
        self.transformation = transformation.to(self.device) # Load the transformation onto device to accelerate training

    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, index) -> torch.Tensor:
        try:
            # Get the path to the specified audio file and its corresponding label
            audio_sample_path = self._get_audio_sample_path(index)
            label = self._get_audio_label(index)

            # Load the signal in as a tensor
            signal, sr = torchaudio.load(audio_sample_path)

            # Load the signal onto the device to accelerate training
            signal = signal.to(self.device)

            # Resample the signal (if its sample rate is not equivalent to the standard rate)
            signal = self._resample_if_necessary(signal, sr)

            # Cut the length of the signal to match the standard number of samples (if necessary)
            signal = self._cut_signal_if_necessary(signal)

            # Pad the length of the signal to match the standard number of samples (if necessary)
            signal = self._pad_signal_if_necessary(signal)

            # Mix the signal down (i.e. convert it to mono) if it is not already in mono
            signal = self._mix_down_if_necessary(signal)

            # Apply the specified transformation to the signal
            signal = self.transformation(signal)
            return signal, label
        except IndexError:
            print(f"No audio sample exists at index {index}")

    def _get_audio_sample_path(self, index: int) -> str:
        # Identify the fold
        fold = f"fold{self.annotations.iloc[index, 5]}"

        # Identify the file name
        file_name = self.annotations.iloc[index, 0]

        # Return the full audio sample path
        return os.path.join(self.audio_file_path, fold, file_name)
    
    def _get_audio_label(self, index: int) -> int:
        # Return the corresponding audio label
        return self.annotations.iloc[index, 6]
    
    def _resample_if_necessary(self, signal: torch.Tensor, sr: int) -> torch.Tensor:
        # If the sample rate is NOT equal to the standard sample rate, resample the signal at the 
        # standard sample rate. Otherwise, just return the signal
        if sr != self.standard_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.standard_sr).to(self.device)
            return resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        # If the signal has more than one channel (i.e. is stereo or greater), take the
        # mean across all columns such that we go from (# of channels, # of samples) -> (1, # of samples)
        if signal.shape[0] > 1:
            return torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _cut_signal_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        # If the number of samples is greater than the standard number of samples,
        # only retain the number of samples up to the standard number of samples. 
        # Otherwise, just return the signal
        if signal.shape[1] > self.standard_num_samples:
            return signal[:, :self.standard_num_samples]
        return signal
    
    def _pad_signal_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        # If the number of samples is less than the standard number of samples,
        # append zeros to the signal (aka pad it) until its length is the standard
        # number of samples.
        # Otherwise, just return the signal
        if signal.shape[1] < self.standard_num_samples:
            pad_amt = self.standard_num_samples - signal.shape[1]
            return torch.nn.functional.pad(input=signal, pad=(0, pad_amt))
        return signal
    
# if __name__ == "__main__":
#     # Constants
#     ANNOTATIONS_FILE_PATH = os.path.join("data", "metadata", "UrbanSound8K.csv")
#     AUDIO_FILE_PATH = os.path.join("data", "audio")
#     STANDARD_SR = 22_050
#     STANDARD_NUM_SAMPLES = 22_050
#     FRAME_SIZE = 1_024
#     HOP_SIZE = 512
#     NUM_MEL_BANDS = 62

#     # Determine which device to use for training
#     device = "cpu" # Train on cpu by default

#     if torch.backends.mps.is_available():
#         device = "mps"

#     print(f"Device to be used: {device}")

#     # Create Mel Spectrogram transform
#     mel_spectrogram = torchaudio.transforms.MelSpectrogram(
#         sample_rate=STANDARD_SR,
#         n_fft=FRAME_SIZE,
#         hop_length=HOP_SIZE,
#         n_mels=NUM_MEL_BANDS
#     )

#     # Load dataset
#     usd = UrbanSoundDataset(
#         annotations_file_path=ANNOTATIONS_FILE_PATH, 
#         audio_file_path=AUDIO_FILE_PATH,
#         device=device,
#         standard_sr=STANDARD_SR,
#         standard_num_samples=STANDARD_NUM_SAMPLES,
#         transformation=mel_spectrogram
#     )

#     # Check that the UrbanSoundDataset class loaded the correct amount of data
#     print(f"The Urban Sound Dataset has {len(usd)} rows!")

#     # Inspect the first signal and label
#     signal, label = usd[0]
#     print(f"The signal is of type {type(signal)} and has dimensions {signal.shape}")
#     print(f"The first audio file class is {label}")