"""Run inference on a single image."""
import argparse
import torch
from torchvision import transforms
from PIL import Image
from models import ResNetSmall

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def predict(image_path, model_path='best_resnet.pt'):
    transform = transforms.Compose([
        transforms.Grayscale(), transforms.Resize((28, 28)),
        transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,)),
    ])
    img = transform(Image.open(image_path)).unsqueeze(0)
    model = ResNetSmall()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(img), dim=1)
        pred = probs.argmax(dim=1).item()
    print(f"Prediction: {CLASS_NAMES[pred]} ({probs[0][pred]:.1%})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', default='best_resnet.pt')
    args = parser.parse_args()
    predict(args.image, args.model)
