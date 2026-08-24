import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from encoder import encoder
from decoder import decoder
def validation_model(validation_dataset,encoder_memory,decoder_memory,device):
    for tool in encoder_memory:
        tool.eval()
    for up_tool in decoder_memory['up']:
        up_tool.eval()
    for blend_tool in decoder_memory['blend']:
        blend_tool.eval()
    decoder_memory['final'].eval()
    total_val_loss = 0.0
    correct_val_pixel = 0
    total_val_pixel = 0
    with torch.no_grad():
        for val_image , val_mask in validation_dataset:
            val_image = val_image.to(device)
            val_mask = val_mask.to(device)
            val_image = val_image/torch.max(val_image)
            x,skip,encoder_memory = encoder(val_image,device,saved_tools=encoder_memory)
            val_prediction,decoder_memory= decoder(x,skip,saved_tools=decoder_memory)
            val_criterion = nn.CrossEntropyLoss()
            val_loss = val_criterion(val_prediction,val_mask)
            total_val_loss += val_loss.item()
            val_predicted = torch.argmax(val_prediction,dim=1)
            correct_val_pixel += (val_predicted == val_mask).sum().item()
            total_val_pixel += val_mask.numel()
        val_loss = total_val_loss/len(validation_dataset)
        val_accuracy = (correct_val_pixel/total_val_pixel)*100
        return val_accuracy,val_loss
def training_model(train_dataset,validation_dataset):
    encoder_memory = None
    decoder_memory = None
    optimizer = None
    h_v_accuracy = []
    h_v_loss = []
    h_t_accuracy = []
    h_t_loss = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n ----- Starting Training loop on {device}----- \n")
    epochs = 0
    current_loss = 100.0
    correct_pixel = 0
    total_pixel = 0
    while current_loss > 0.000001:
        epochs += 1
        total_epoch_loss = 0.0
        for batch_image , batch_mask in train_dataset:
            batch_image = batch_image.to(device)
            batch_mask = batch_mask.to(device)
            batch_image = batch_image/torch.max(batch_image)
            x,skip,encoder_memory = encoder(batch_image,device,saved_tools=encoder_memory)
            prediction , decoder_memory = decoder(x,skip,saved_tools=decoder_memory)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(prediction,batch_mask)
            if optimizer is None:
                all_weight = []
                for tool in encoder_memory:
                    for parn in tool.parameters():
                        all_weight.append(parn)
                for i in range(len(decoder_memory['up'])):
                    for parn in decoder_memory['up'][i].parameters():
                        all_weight.append(parn)
                for i in range(len(decoder_memory['blend'])):
                    for parn in decoder_memory['blend'][i].parameters():
                        all_weight.append(parn)
                for parn in decoder_memory['final'].parameters():
                    all_weight.append(parn)
                optimizer = optim.Adam(all_weight,lr=0.01)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
            predicted_mask = torch.argmax(prediction,dim=1)
            correct_pixel+=(predicted_mask==batch_mask).sum().item()
            total_pixel += batch_mask.numel()
        val_accuracy,val_loss = validation_model(validation_dataset,encoder_memory,decoder_memory,device)
        for tool in encoder_memory:
            tool.train()
        for up_tool in decoder_memory['up']:
            up_tool.train()
        for blend_tool in decoder_memory['blend']:
            blend_tool.train()
        decoder_memory['final'].train()
        current_loss = round(total_epoch_loss/len(train_dataset),4)
        training_accuracy = (correct_pixel/total_pixel)*100
        h_t_accuracy.append(training_accuracy)
        h_t_loss.append(current_loss)
        h_v_accuracy.append(val_accuracy)
        h_v_loss.append(val_loss)
    print(f"Epoch:{epochs}| Loss:{current_loss:.4f} |Accuracy:{training_accuracy:.2f}%")
    print(f"Validation Loss:{val_loss:.4f} |Accuracy:{val_accuracy:.2f}%")
    print("\n ----- model training complete ----- \n")
    val_image_batch, val_mask_batch = next(iter(validation_dataset))
    val_image_batch = val_image_batch.to(device)
    val_mask_batch = val_mask_batch.to(device)
    norm_val_image = (val_image_batch / torch.max(val_image_batch))
    for tool in encoder_memory:
        tool.eval()
    for up_tool in decoder_memory['up']:
        up_tool.eval()
    for blend_tool in decoder_memory['blend']:
        blend_tool.eval()
    decoder_memory['final'].eval()
    with torch.no_grad():
        x, skip, _ = encoder(norm_val_image,device,saved_tools=encoder_memory)
        val_prediction, _ = decoder(x,skip,saved_tools=decoder_memory)
        predicted_val_mask = torch.argmax(val_prediction,dim=1)
    visual_image = (val_image_batch[0, 0, :, :].cpu().detach().numpy())
    visual_mask = (val_mask_batch[0].cpu().detach().numpy())
    visual_prediction = (predicted_val_mask[0].cpu().detach().numpy())
    fig, axes = plt.subplots(1,3,figsize=(15, 5))
    axes[0].imshow(visual_image,cmap="gray")
    axes[0].set_title("Validation Image - Band 0",fontsize=12,fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(visual_mask,cmap="jet",vmin=0,vmax=16)
    axes[1].set_title("True Target Mask",fontsize=12,fontweight="bold")
    axes[1].axis("off")
    axes[2].imshow(visual_prediction,cmap="jet",vmin=0,vmax=16)
    axes[2].set_title("AI Validation Prediction",fontsize=12,fontweight="bold")
    axes[2].axis("off")
    plt.tight_layout()
    print("\nDisplaying final validation prediction...")
    plt.show()
    for tool in encoder_memory:
        tool.train()
    for up_tool in decoder_memory['up']:
        up_tool.train()
    for blend_tool in decoder_memory['blend']:
        blend_tool.train()
    decoder_memory['final'].train()
    model_save_path = "weights.pth"
    torch.save({
        'encoder_state': encoder_memory.state_dict(),
        'decoder_state': decoder_memory.state_dict()
    }, model_save_path)
    print(f"Success! Model weights saved to {model_save_path}")
    np.savez(
        "training_history.npz",
        training_accuracy=np.array(h_t_accuracy),
        training_loss=np.array(h_t_loss),
        validation_accuracy=np.array(h_v_accuracy),
        validation_loss=np.array(h_v_loss)
    )
    return h_t_accuracy,h_t_loss,h_v_accuracy,h_v_loss
#completed