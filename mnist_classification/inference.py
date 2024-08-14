import torch
from train import download_mnist_data, FeedForwardNet
from typing import List, Tuple

class_mapping = [str(i) for i in range(10)]

def predict(model: FeedForwardNet, input: torch.tensor, label: str, class_mapping: List[str]) -> Tuple[str]:
    # Change the model to evaluation mode
    model.eval()

    # Run inference without calculating gradients to speed up the process
    with torch.no_grad():
        predictions = model(input) # Returns a torch.tensor with shape (1, 10)
        predicted_index = predictions[0].argmax(0)
        predicted_class = class_mapping[predicted_index]
        expected_class = class_mapping[label]
    
    return predicted_class, expected_class

if __name__ == "__main__":
    # Load back the model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth", weights_only=True)
    feed_forward_net.load_state_dict(state_dict=state_dict)
    
    # Download the test data
    _, test_data = download_mnist_data()

    # Get a sample for inference
    input, label = test_data[0][0], test_data[0][1]

    # Make an inference
    predicted, expected = predict(model=feed_forward_net, input=input, label=label, class_mapping=class_mapping)

    print(f"Predicted: {predicted} | Expected: {expected}")