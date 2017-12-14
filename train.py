"""Train fashion classifier."""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import FashionCNN, ResNetSmall

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = model(images).max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['cnn', 'resnet'], default='resnet')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    train_data = datasets.FashionMNIST('data', train=True, download=True, transform=get_transforms(True))
    test_data = datasets.FashionMNIST('data', train=False, transform=get_transforms(False))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    model = (ResNetSmall() if args.model == 'resnet' else FashionCNN()).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    for epoch in range(args.epochs):
        loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'best_{args.model}.pt')
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
    print(f"\nBest test accuracy: {best_acc:.4f}")
# TODO: add per-class accuracy breakdown

if __name__ == '__main__':
    main()
