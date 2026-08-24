import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from data_split import split
from training import training_model
from testing import testing
from encoder import encoder
from decoder import decoder
st.set_page_config(page_title="U-Net Indian Pines Segmentation",page_icon="",layout="wide")
st.title(" U-Net Image Segmentation")
st.markdown(
    """
    This dashboard displays the complete U-Net workflow:
    **Original Image -> Ground Truth -> Patches -> Training -> Validation -> Testing -> Full Image Reconstruction**
    """
)
st.divider()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    st.success(" Model is running on CUDA / GPU")
else:
    st.warning("⚠ Model is running on CPU")
@st.cache_data
def load_data():
    img = np.load("indianpinearray.npy")
    mask = np.load("IPgt.npy")
    return img, mask
img, original_mask = load_data()
original_image_shape = img.shape[:2]
st.header("1⃣ Original Indian Pines Dataset")
col1, col2 = st.columns(2)
rgb_image = img[:, :, [30, 20, 10]]
rgb_image = rgb_image.astype(np.float32)
rgb_min = rgb_image.min()
rgb_max = rgb_image.max()
if rgb_max > rgb_min:
    rgb_image = (rgb_image - rgb_min) / (rgb_max - rgb_min)
with col1:
    st.subheader("Original Indian Pines Image")
    st.image(rgb_image,use_container_width=True)
with col2:
    st.subheader("Original Ground Truth Mask")
    fig, ax = plt.subplots()
    ax.imshow(original_mask,cmap="jet",vmin=0,vmax=16)
    ax.axis("off")
    st.pyplot(fig)
st.header("Dataset Information")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Image Height",img.shape[0])
c2.metric("Image Width",img.shape[1])
c3.metric("Spectral Bands",img.shape[2])
c4.metric("Classes",17)
st.write(f"Original image shape: `{img.shape}`")
st.write(f"Original mask shape: `{original_mask.shape}`")
image_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
pad_image = torch.nn.functional.pad(image_tensor,(7, 8, 7, 8))
mask = original_mask.astype(np.float32)
mask_tensor = torch.from_numpy(mask)
pad_mask = torch.nn.functional.pad(mask_tensor,(7, 8, 7, 8))
image_patches, mask_patches = split(pad_image,pad_mask)
st.header("Image Patches")
p1, p2, p3 = st.columns(3)
p1.metric("Total Patches",image_patches.shape[0])
p2.metric("Patch Height",image_patches.shape[2])
p3.metric("Patch Width",image_patches.shape[3])
st.write(f"Image patches shape: `{image_patches.shape}`")
st.write(f"Mask patches shape: `{mask_patches.shape}`")
dataset = TensorDataset(image_patches,mask_patches)
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
validation_size = int(0.19 * dataset_size)
test_size = (dataset_size-train_size-validation_size)
generator = torch.Generator().manual_seed(42)
train_dataset, validation_dataset, test_dataset = random_split(dataset,[train_size,validation_size,test_size],generator=generator)
train_loader = DataLoader(train_dataset,batch_size=5,shuffle=True)
validation_loader = DataLoader(validation_dataset,batch_size=4,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=4,shuffle=False)
st.header("4⃣ Dataset Split")
c1, c2, c3 = st.columns(3)
c1.metric("Training",train_size)
c2.metric("Validation",validation_size)
c3.metric("Testing",test_size)
st.header("5⃣ U-Net Training")
st.write(
"Click the button below to train the U-Net using the same "
"training function used by your original project."
)
if "training_done" not in st.session_state:
    st.session_state.training_done = False
if st.button(" Start Training",type="primary"):
    with st.spinner("Training U-Net... Please wait."):
        (train_accuracy,train_loss,validation_accuracy,validation_loss) = training_model(train_loader,validation_loader)
    st.session_state.training_done = True
    st.success("Training completed successfully!")
    np.savez("training_history.npz",training_accuracy=np.array(train_accuracy),training_loss=np.array(train_loss),validation_accuracy=np.array(validation_accuracy),validation_loss=np.array(validation_loss))
st.header("6⃣ Training Graphs")
if st.session_state.training_done:
    history = np.load("training_history.npz")
    train_loss = history["training_loss"]
    validation_loss = history["validation_loss"]
    train_accuracy = history["training_accuracy"]
    validation_accuracy = history["validation_accuracy"]
    st.subheader("Training and Validation Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Accuracy", f"{train_accuracy[-1]:.2f}%")
        st.metric("Training Loss", f"{train_loss[-1]:.4f}")
    with col2:
        st.metric("Validation Accuracy", f"{validation_accuracy[-1]:.2f}%")
        st.metric("Validation Loss", f"{validation_loss[-1]:.4f}")
    st.subheader("Loss Curve")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(train_loss,label="Training Loss")
        ax.plot(validation_loss,label="Validation Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        st.subheader("Accuracy Curve")
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(train_accuracy,label="Training Accuracy")
        ax.plot(validation_accuracy,label="Validation Accuracy")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Training and Validation Accuracy")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
else:
    if __import__("os").path.exists("training_history.npz"):
        history = np.load("training_history.npz")
        fig, ax = plt.subplots()
        ax.plot(history["training_loss"],label="Training Loss")
        ax.plot(history["validation_loss"],label="Validation Loss")
        ax.set_title("Loss Curve")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        fig, ax = plt.subplots()
        ax.plot(history["training_accuracy"],label="Training Accuracy")
        ax.plot(history["validation_accuracy"],label="Validation Accuracy")
        ax.set_title("Accuracy Curve")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("Training graphs will appear here after training.")
st.header("7⃣ Validation Prediction")
if st.button(" Run Validation Prediction"):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    checkpoint = torch.load("weights.pth",map_location=device)
    sample_image, sample_mask = next(iter(validation_loader))
    sample_image = sample_image.to(device)
    sample_mask = sample_mask.to(device)
    sample_image = (sample_image / torch.max(sample_image))
    encoder_memory = None
    decoder_memory = None
    with torch.no_grad():
        x, skip_connections, encoder_memory = encoder(sample_image,device,saved_tools=None)
        prediction, decoder_memory = decoder(x,skip_connections,saved_tools=None)
    encoder_memory.load_state_dict(checkpoint["encoder_state"])
    decoder_memory.load_state_dict(checkpoint["decoder_state"])
    encoder_memory.eval()
    for layer in decoder_memory["up"]:
        layer.eval()
    for layer in decoder_memory["blend"]:
        layer.eval()
    decoder_memory["final"].eval()
    with torch.no_grad():
        x, skip_connections, _ = encoder(sample_image,device,saved_tools=encoder_memory)
        prediction, _ = decoder(x,skip_connections,saved_tools=decoder_memory)
        predicted_mask = torch.argmax(prediction,dim=1)
    visual_image = (sample_image[0, 0].cpu().numpy())
    visual_mask = (sample_mask[0].cpu().numpy())
    visual_prediction = (predicted_mask[0].cpu().numpy())
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(visual_image,caption="Validation Image - Band 0",use_container_width=True)
    with col2:
        fig, ax = plt.subplots()
        ax.imshow(visual_mask,cmap="jet",vmin=0,vmax=16)
        ax.axis("off")
        ax.set_title("True Target Mask")
        st.pyplot(fig)
    with col3:
        fig, ax = plt.subplots()
        ax.imshow(visual_prediction,cmap="jet",vmin=0,vmax=16)
        ax.axis("off")
        ax.set_title("AI Validation Prediction")
        st.pyplot(fig)
st.header("8⃣ Testing")
if st.button(" Run Testing"):
    (test_loss,test_accuracy,visual_image,visual_mask,visual_prediction) = testing(test_loader)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Test Loss",f"{test_loss:.4f}")
    with c2:
        st.metric("Test Accuracy",f"{test_accuracy:.2f}%")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(visual_image,caption="Test Image - Band 0",use_container_width=True)
    with col2:
        fig, ax = plt.subplots()
        ax.imshow(
            visual_mask,
            cmap="jet",
            vmin=0,
            vmax=16
        )
        ax.axis("off")
        ax.set_title(
            "True Target Mask"
        )
        st.pyplot(fig)
    with col3:
        fig, ax = plt.subplots()
        ax.imshow(
            visual_prediction,
            cmap="jet",
            vmin=0,
            vmax=16
        )
        ax.axis("off")
        ax.set_title(
            "U-Net Prediction"
        )
        st.pyplot(fig)
st.header("9⃣ Full Image Reconstruction")
if st.button(" Reconstruct Complete 145 × 145 Map",type="primary"):
    checkpoint = torch.load("weights.pth",map_location=device)
    reconstruction_loader = DataLoader(TensorDataset(image_patches,mask_patches),batch_size=5,shuffle=False)
    sample_image, _ = next(iter(reconstruction_loader))
    sample_image = sample_image.to(device)
    sample_image = (sample_image/ torch.max(sample_image))
    encoder_memory = None
    decoder_memory = None
    with torch.no_grad():
        x, skip_connections, encoder_memory = encoder(sample_image,device,saved_tools=None)
        prediction, decoder_memory = decoder(x,skip_connections,saved_tools=None)
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
    padded_height = (patches_per_side* patch_size)
    padded_width = (patches_per_side* patch_size)
    reconstructed_prediction = np.zeros((padded_height,padded_width),dtype=np.int64)
    reconstructed_ground_truth = np.zeros((padded_height,padded_width),dtype=np.int64)
    patch_index = 0
    with torch.no_grad():
        for batch_image, batch_mask in reconstruction_loader:
            batch_image = batch_image.to(device)
            batch_image = (batch_image / torch.max(batch_image))
            x, skip_connections, _ = encoder(batch_image,device,saved_tools=encoder_memory)
            prediction, _ = decoder(x,skip_connections,saved_tools=decoder_memory)
            predicted_mask = torch.argmax(prediction,dim=1)
            predicted_mask = (predicted_mask.cpu().numpy())
            batch_mask = (batch_mask.numpy())
            for i in range(
                predicted_mask.shape[0]):
                row = (patch_index // patches_per_side)
                column = (patch_index % patches_per_side)
                y_start = (row * patch_size)
                y_end = (y_start + patch_size)
                x_start = (column * patch_size)
                x_end = (x_start + patch_size)
                reconstructed_prediction[y_start:y_end,x_start:x_end] = predicted_mask[i]
                reconstructed_ground_truth[y_start:y_end,x_start:x_end] = batch_mask[i]
                patch_index += 1
    reconstructed_prediction = (reconstructed_prediction[7:7 + original_image_shape[0],7:7 + original_image_shape[1]])
    reconstructed_ground_truth = (reconstructed_ground_truth[7:7 + original_image_shape[0],7:7 + original_image_shape[1]])
    reconstructed_accuracy = (np.mean(reconstructed_prediction== reconstructed_ground_truth)*100)
    st.metric("Full Image Reconstruction Accuracy",f"{reconstructed_accuracy:.2f}%")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.imshow(reconstructed_ground_truth,cmap="jet",vmin=0,vmax=16)
        ax.axis("off")
        ax.set_title("Complete Ground Truth Mask")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.imshow(reconstructed_prediction,cmap="jet",vmin=0,vmax=16)
        ax.axis("off")
        ax.set_title("Complete U-Net Predicted Mask")
        st.pyplot(fig)
    st.subheader("Prediction Error Map")
    error_map = (reconstructed_prediction!= reconstructed_ground_truth)
    fig, ax = plt.subplots()
    ax.imshow(error_map,cmap="gray")
    ax.axis("off")
    ax.set_title("Incorrectly Classified Pixels")
    st.pyplot(fig)
st.divider()
st.caption("U-Net Image Segmentation | Indian Pines Dataset | PyTorch + Streamlit")