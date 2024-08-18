import torch
import torchaudio
from cnn import CNN
from train_cnn import ANNOTATIONS_FILE_PATH, AUDIO_FILE_PATH, FRAME_SIZE, HOP_SIZE, NUM_MEL_BANDS, STANDARD_NUM_SAMPLES, STANDARD_SR
from urban_sound_dataset import UrbanSoundDataset
from typing import List, Tuple

class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

def predict(model: UrbanSoundDataset, input: torch.tensor, label: str, class_mapping: List[str]) -> Tuple[str]:
    # Change the model to evaluation mode
    model.eval()

    # Run inference without calculating gradients to speed up the process
    with torch.no_grad():
        predictions = model(input) # Returns the logits, which are a torch.tensor with shape (1, 10)
        predictions = torch.softmax(input=predictions, dim=1) # Get the probabilities
        predicted_index = predictions[0].argmax(0)
        predicted_class = class_mapping[predicted_index]
        expected_class = class_mapping[label]
    
    return predicted_class, expected_class

if __name__ == "__main__":
    # Get device
    device = "cpu"

    if torch.backends.mps.is_available():
        device = "mps"

    print(f"Using {device}")

    # Load back the model
    cnn = CNN().to(device)
    state_dict = torch.load("urban_sound_classifier.pth", weights_only=True)
    cnn.load_state_dict(state_dict=state_dict)
    
    # Create Mel Spectrogram transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=STANDARD_SR,
        n_fft=FRAME_SIZE,
        hop_length=HOP_SIZE,
        n_mels=NUM_MEL_BANDS
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

    # Get a sample for inference
    input, label = usd[0][0], usd[0][1]
    input.unsqueeze_(0)

    # Make an inference
    predicted, expected = predict(model=cnn, input=input, label=label, class_mapping=class_mapping)

    print(f"Predicted: {predicted} | Expected: {expected}")