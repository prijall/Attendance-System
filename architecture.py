import torch.nn as nn
import os, torch
from PIL import Image
from torchvision import transforms 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir=root_dir
        self.image_paths=[]
        self.labels=[]
        
        #traversing the directories:
        for label, user_folder in enumerate(os.listdir(root_dir)):
            user_folder_path=os.path.join(root_dir, user_folder)
            if os.path.isdir(user_folder_path):
                for img_file in os.listdir(user_folder_path):
                    img_path=os.path.join(user_folder_path, img_file)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path=self.image_paths[idx]
        image=Image.open(img_path).convert('RGB')
        label=self.labels[idx]

        if self.transform:
            image=self.transform(image)
        
        return image, label
    
#@ transformation:
IMAGE_SIZE=224 #for resnet 50
transform=transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) #the values are from image-net dataset training
])

#@ Loading the dataset:
train_dataset=FaceDataset(os.path.join('Dataset', 'train')) 
test_dataset=FaceDataset(os.path.join('Dataset', 'test')) 

train_dl=DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dl=DataLoader(test_dataset, batch_size=32, shuffle=True)
