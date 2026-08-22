import torch
def split(img,mask,patch_size=32):
    image_list = []
    mask_list= []
    image_shape = img.shape[2]
    if(image_shape%2==0):
        num_patches = image_shape//patch_size
        for row in range(num_patches):
            for column in range(num_patches):
                y_start = row*patch_size
                y_end = (row+1)*patch_size
                x_start = column*patch_size
                x_end = (column+1)*patch_size
                img_patch = img[:,y_start:y_end,x_start:x_end]
                mask_patch = mask[y_start:y_end,x_start:x_end]
                image_list.append(img_patch)
                mask_list.append(mask_patch)
        image_patches = torch.stack(image_list)
        mask_patches = torch.stack(mask_list)
        mask_patches = mask_patches.long()
        return image_patches, mask_patches