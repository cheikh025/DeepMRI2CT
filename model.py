import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define a simple convolutional block
def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# Define a simple Convolutional Neural Network for MRI to CT image translation
class MRIToCTModel(nn.Module):
    def __init__(self):
        super(MRIToCTModel, self).__init__()
        # Encoder: Compressing the MRI image into feature representations
        self.encoder = nn.Sequential(
            conv_block(1, 64),  # 256x256 -> 256x256
            nn.MaxPool2d(2),   # 256x256 -> 128x128
            conv_block(64, 128),
            nn.MaxPool2d(2),   # 128x128 -> 64x64
            conv_block(128, 256),
            nn.MaxPool2d(2),   # 64x64 -> 32x32
        )
        # Decoder: Reconstructing the CT image from the feature representations
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # 32x32 -> 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),   # 64x64 -> 128x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 2, stride=2),     # 128x128 -> 256x256
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

def train(model, dataloader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        for mri_images, ct_images, masks in dataloader:
            mri_images, ct_images, masks = mri_images.to(device), ct_images.to(device), masks.to(device)
            optimizer.zero_grad()
            pred_ct_images = model(mri_images)
            masked_pred_ct = pred_ct_images * masks
            masked_ct_images = ct_images * masks
            loss = criterion(masked_pred_ct, masked_ct_images)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")