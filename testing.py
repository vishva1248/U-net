import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from encoder import encoder
from decoder import decoder
def testing(test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-----TESTING-----")
    print(f"-----Testing on device: {device}-----")
    model_path = "weights.pth"
    try:
        checkpoint = torch.load(model_path,map_location=device)
    except FileNotFoundError:
        print("\nERROR: weights.pth not found.")
        print("Please train the model first.")
        return
    print("Weights loaded successfully.")
    encoder_memory = None
    decoder_memory = None
    sample_image, sample_mask = next(iter(test_dataset))
    sample_image = sample_image.to(device)
    sample_image = sample_image / torch.max(sample_image)
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
    criterion = nn.CrossEntropyLoss()
    total_test_loss = 0.0
    correct_pixel = 0
    total_pixel = 0
    visual_done = False
    with torch.no_grad():
        for batch_image, batch_mask in test_dataset:
            batch_image = batch_image.to(device)
            batch_mask = batch_mask.to(device)
            batch_image = batch_image / torch.max(batch_image)
            x, skip_connections, _ = encoder(batch_image,device,saved_tools=encoder_memory)
            prediction, _ = decoder(x,skip_connections,saved_tools=decoder_memory)
            loss = criterion(prediction,batch_mask)
            total_test_loss += loss.item()
            predicted_mask = torch.argmax(prediction,dim=1)
            correct_pixel += (predicted_mask == batch_mask).sum().item()
            total_pixel += batch_mask.numel()
            if not visual_done:
                visual_image = (batch_image[0, 0, :, :].cpu().detach().numpy())
                visual_mask = (batch_mask[0].cpu().detach().numpy())
                visual_prediction = (predicted_mask[0].cpu().detach().numpy())
                fig, axes = plt.subplots(1,3,figsize=(15, 5))
                axes[0].imshow(visual_image,cmap="gray")
                axes[0].set_title("Test Image - Band 0",fontsize=12,fontweight="bold")
                axes[0].axis("off")
                axes[1].imshow(visual_mask,cmap="jet",vmin=0,vmax=16)
                axes[1].set_title("True Target Mask",fontsize=12,fontweight="bold")
                axes[1].axis("off")
                axes[2].imshow(visual_prediction,cmap="jet",vmin=0,vmax=16)
                axes[2].set_title("U-Net Prediction",fontsize=12,fontweight="bold")
                axes[2].axis("off")
                plt.tight_layout()
                plt.show()
                visual_done = True
    test_loss = (total_test_loss / len(test_dataset))
    test_accuracy = (correct_pixel / total_pixel)*100
    print("-----TESTING RESULTS-----")
    print(f"Test Loss : {test_loss:.4f}|Test Accuracy : {test_accuracy:.2f}%")
    print("-----TESTING COMPLETE------")
#completed