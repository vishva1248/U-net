import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader, TensorDataset , random_split
from data_split import split
from training import training_model
from plot import plot
from testing import testing
from encoder import encoder
from decoder import decoder
def reconstruct_full_prediction(image_patches, mask_patches, original_image_shape):
    answer = input("type yes to reconstruct full prediction:- ")
    if answer == "yes":
        """
            Run the trained U-Net on all 25 patches and reconstruct
            the complete predicted segmentation mask.
            """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\n-----FULL IMAGE RECONSTRUCTION-----")
        print(f"-----Reconstruction on device: {device}-----")
        try:
            checkpoint = torch.load("weights.pth", map_location=device)
        except FileNotFoundError:
            print("\nERROR: weights.pth not found.")
            print("Please train the model first.")
            return
        print("Weights loaded successfully.")
        reconstruction_dataset = TensorDataset(image_patches, mask_patches)
        reconstruction_loader = DataLoader(reconstruction_dataset, batch_size=5, shuffle=False)
        sample_image, _ = next(iter(reconstruction_loader))
        sample_image = sample_image.to(device)
        sample_image = sample_image / torch.max(sample_image)
        encoder_memory = None
        decoder_memory = None
        with torch.no_grad():
            x, skip_connections, encoder_memory = encoder(sample_image, device, saved_tools=None)
            prediction, decoder_memory = decoder(x, skip_connections, saved_tools=None)
        encoder_memory.load_state_dict(checkpoint["encoder_state"])
        decoder_memory.load_state_dict(checkpoint["decoder_state"])
        encoder_memory.eval()
        for layer in decoder_memory["up"]:
            layer.eval()
        for layer in decoder_memory["blend"]:
            layer.eval()
        decoder_memory["final"].eval()
        patch_size = image_patches.shape[-1]
        total_patches = image_patches.shape[0]
        patches_per_side = int(np.sqrt(total_patches))
        if patches_per_side ** 2 != total_patches:
            raise ValueError(f"Expected a square patch grid, "f"but found {total_patches} patches.")
        padded_height = patches_per_side * patch_size
        padded_width = patches_per_side * patch_size
        print(f"Total patches: {total_patches}")
        print(f"Patch size: {patch_size} x {patch_size}")
        print(f"Patch grid: "f"{patches_per_side} x {patches_per_side}")
        reconstructed_prediction = np.zeros((padded_height, padded_width), dtype=np.int64)
        reconstructed_ground_truth = np.zeros((padded_height, padded_width), dtype=np.int64)
        patch_index = 0
        with torch.no_grad():
            for batch_image, batch_mask in reconstruction_loader:
                batch_image = batch_image.to(device)
                batch_image = (batch_image / torch.max(batch_image))
                x, skip_connections, _ = encoder(batch_image, device, saved_tools=encoder_memory)
                prediction, _ = decoder(x, skip_connections, saved_tools=decoder_memory)
                predicted_mask = torch.argmax(prediction, dim=1)
                predicted_mask = (predicted_mask.cpu().numpy())
                batch_mask = batch_mask.numpy()
                for i in range(predicted_mask.shape[0]):
                    row = patch_index // patches_per_side
                    column = patch_index % patches_per_side
                    y_start = row * patch_size
                    y_end = y_start + patch_size
                    x_start = column * patch_size
                    x_end = x_start + patch_size
                    reconstructed_prediction[y_start:y_end,x_start:x_end] = predicted_mask[i]
                    reconstructed_ground_truth[y_start:y_end,x_start:x_end] = batch_mask[i]
                    patch_index += 1
        original_height, original_width = original_image_shape
        reconstructed_prediction = reconstructed_prediction[7:7 + original_height, 7:7 + original_width]
        reconstructed_ground_truth = reconstructed_ground_truth[7:7 + original_height, 7:7 + original_width]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(reconstructed_ground_truth, cmap="jet", vmin=0, vmax=16)
        axes[0].set_title("Complete Ground Truth Mask", fontsize=12, fontweight="bold")
        axes[0].axis("off")
        axes[1].imshow(reconstructed_prediction, cmap="jet", vmin=0, vmax=16)
        axes[1].set_title("Complete U-Net Predicted Mask", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()
        print("-----FULL IMAGE RECONSTRUCTION COMPLETE-----")
    else:
        print("No prediction was made.")
        exit()
img = np.load("indianpinearray.npy")
mask = np.load("IPgt.npy")
print("Original image shape :", img.shape)
print("Original mask shape  :", mask.shape)
original_image_shape = img.shape[:2]
rgb_image = img[:, :, [30, 20, 10]]
rgb_image = rgb_image.astype(np.float32)
rgb_min = rgb_image.min()
rgb_max = rgb_image.max()
if rgb_max > rgb_min:
    rgb_image = (rgb_image - rgb_min) / (rgb_max - rgb_min)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(rgb_image)
plt.title("Original Indian Pines Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="jet")
plt.title("Original Ground Truth Mask")
plt.axis("off")
plt.tight_layout()
plt.show()
image_tensor = torch.from_numpy(img).permute(2,0,1).float()
pad_image = f.pad(image_tensor,(7,8,7,8))
mask = mask.astype(np.float32)
mask_tensor = torch.from_numpy(mask)
pad_mask = f.pad(mask_tensor,(7,8,7,8))
image_patches , mask_patches = split(pad_image,pad_mask)
print("Final image patches shape :-",image_patches.shape)
print("Final mask patches shape :-",mask_patches.shape)
my_dataset = TensorDataset(image_patches,mask_patches)
dataset_size = len(my_dataset)
train_size = int(0.7 * dataset_size)
validation_size = int(0.19 * dataset_size)
test_size = (dataset_size - train_size - validation_size)
generator = torch.Generator().manual_seed(42)
train_dataset, validation_dataset, test_dataset = random_split(my_dataset,[train_size, validation_size, test_size],generator=generator)
train_dataset = DataLoader(train_dataset,batch_size=5,shuffle=True)
validation_dataset = DataLoader(validation_dataset,batch_size=4,shuffle=False)
test_dataset = DataLoader(test_dataset,batch_size=4,shuffle=False)
print(f"Training on {train_size} patches , Validation on {validation_size} , Testing on {test_size} patches")
mode = input("\nEnter mode (training/testing): ").strip().lower()
if mode == "training":
    h_t_accuracy ,h_t_loss , h_v_accuracy , h_v_loss = training_model(train_dataset,validation_dataset)
    plot(h_t_loss,h_t_accuracy,h_v_loss,h_v_accuracy)
    testing(test_dataset)
    reconstruct_full_prediction(image_patches,mask_patches,original_image_shape)
elif mode == "testing":
    testing(test_dataset)
    reconstruct_full_prediction(image_patches,mask_patches,original_image_shape)
else:
    print("\nInvalid mode.")
    print("Please enter either 'training' or 'testing'.")
    exit()
#completed