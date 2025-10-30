"""
Gradio app for CIFAR-10 image classification using ResNet18

Deploy this to Hugging Face Spaces
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

# Load model
device = torch.device('cpu')  # Hugging Face Spaces uses CPU
model = torch.jit.load('models/resnet18_deployment.pt', map_location=device)
model.eval()

# CIFAR-10 classes
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    """Predict image class"""
    # Preprocess
    img = Image.fromarray(image).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    # Format results
    results = {CLASSES[i]: float(probs[i]) for i in range(10)}
    return results

# Create interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="CIFAR-10 Image Classifier",
    description="Upload an image to classify it into one of 10 categories. Uses ResNet18 fine-tuned on CIFAR-10 (93.55% accuracy).",
    examples=[
        ["examples/cat.jpg"],
        ["examples/airplane.jpg"],
        ["examples/ship.jpg"]
    ]
)

if __name__ == "__main__":
    demo.launch()
