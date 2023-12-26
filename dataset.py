import os
import torch
import nibabel as nib
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def save_middle_slices(input_folder, output_folder, patient_id):
    # File paths
    mri_path = os.path.join(input_folder, patient_id, 'mr.nii')
    ct_path = os.path.join(input_folder, patient_id, 'ct.nii')
    mask_path = os.path.join(input_folder, patient_id, 'mask.nii')

    # Load images
    mri_image = nib.load(mri_path).get_fdata()
    ct_image = nib.load(ct_path).get_fdata()
    mask_image = nib.load(mask_path).get_fdata()

    # Calculate the middle index along each dimension
    mid_x = mri_image.shape[0] // 2
    mid_y = mri_image.shape[1] // 2
    mid_z = mri_image.shape[2] // 2

    # Extract the middle slices
    axial_slices = {
        'mri': mri_image[:, :, mid_z],
        'ct': ct_image[:, :, mid_z],
        'mask': mask_image[:, :, mid_z],
    }
    coronal_slices = {
        'mri': mri_image[:, mid_y, :],
        'ct': ct_image[:, mid_y, :],
        'mask': mask_image[:, mid_y, :],
    }
    sagittal_slices = {
        'mri': mri_image[mid_x, :, :],
        'ct': ct_image[mid_x, :, :],
        'mask': mask_image[mid_x, :, :],
    }

    # Create directories if they don't exist
    middle_slice_folder = os.path.join(output_folder, patient_id, 'middle_slices')
    os.makedirs(middle_slice_folder, exist_ok=True)

    # Save the middle slices
    for slice_type, slices in zip(['axial', 'coronal', 'sagittal'], [axial_slices, coronal_slices, sagittal_slices]):
        for modality, slice_data in slices.items():
            plt.imsave(os.path.join(middle_slice_folder, f'{modality}_{slice_type}.png'), slice_data, cmap='gray')


def plotImage(image_data):
    mid_x = image_data.shape[0] // 2
    mid_y = image_data.shape[1] // 2
    mid_z = image_data.shape[2] // 2

    # Create subplots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Axial slice (X-Y plane)
    ax[0].imshow(image_data[:, :, mid_z], cmap='gray')
    ax[0].set_title('Axial Slice')

    # Coronal slice (X-Z plane)
    ax[1].imshow(image_data[:, mid_y, :], cmap='gray')
    ax[1].set_title('Coronal Slice')

    # Sagittal slice (Y-Z plane)
    ax[2].imshow(image_data[mid_x, :, :], cmap='gray')
    ax[2].set_title('Sagittal Slice')

    # Display the plots
    plt.show()


class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Assuming subdirectories are structured as patient_id/middle_slices/...
        self.patients = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
    def __len__(self):
        # Assuming each patient has 3 middle slices for MRI, CT, and mask
        return len(self.patients) * 3

    def __getitem__(self, idx):
        patient_id = self.patients[idx // 3]  # Integer division to group every three slices
        slice_type = ['axial', 'coronal', 'sagittal'][idx % 3]  # Modulo to cycle through slice types
        
        mri_path = os.path.join(self.root_dir, patient_id, 'middle_slices', f'mri_{slice_type}.png')
        ct_path = os.path.join(self.root_dir, patient_id, 'middle_slices', f'ct_{slice_type}.png')
        mask_path = os.path.join(self.root_dir, patient_id, 'middle_slices', f'mask_{slice_type}.png')

        mri_image = Image.open(mri_path).convert('L')  # Convert to grayscale
        ct_image = Image.open(ct_path).convert('L')  # Convert to grayscale
        mask_image = Image.open(mask_path).convert('L')  # Convert to grayscale

        if self.transform:
            mri_image = self.transform(mri_image)
            ct_image = self.transform(ct_image)
            mask_image = self.transform(mask_image)

        return mri_image, ct_image, mask_image

def load_medical_images(root_dir, batch_size=1, shuffle=True):
    """
    Load and transform medical images from a specified directory.
    """
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        # Add other transformations here
    ])

    # Instantiate the dataset
    medical_dataset = MedicalImageDataset(root_dir=root_dir, transform=transform)

    # Create and return a DataLoader
    return DataLoader(medical_dataset, batch_size=batch_size, shuffle=shuffle)