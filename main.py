import torch
from model import train, MRIToCTModel
from dataset import MedicalImageDataset, load_medical_images
from torch.utils.data import DataLoader
from torchvision import transforms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = MRIToCTModel().to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    

    # Create DataLoader
    dataloader = load_medical_images(
        root_dir='new_dataset/brain',
        batch_size=4,
        shuffle=True)

    # Train the model
    train(model, dataloader, criterion, optimizer, num_epochs=100, device=device)

if __name__ == "__main__":
    main()
