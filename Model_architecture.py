import torch.nn as nn
import os, torch
from PIL import Image
from torchvision import transforms 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score

#@ For accessing GPU is present (Cuda stands for Compute Unified Device Architecture and does parallel processing in GPU level)
device='cuda' if torch.cuda.is_available() else 'cpu'

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir=root_dir
        self.transform=transform
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
        
        return image, torch.tensor(label)
    
#@ transformation:
IMAGE_SIZE=224 #standard as used by image-net dataset
transform=transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) #the values are from image-net dataset training
])

#@ Loading the dataset:
train_dataset=FaceDataset(os.path.join('Dataset', 'train'), transform=transform) 
test_dataset=FaceDataset(os.path.join('Dataset', 'test'), transform=transform) 

train_dl=DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dl=DataLoader(test_dataset, batch_size=32, shuffle=True)


#@ CNN Model:
class FaceRecognitionCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionCNN, self).__init__()

        #defining CNN Layers:
        self.conv1=nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2=nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3=nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.fc1=nn.Linear(128*28*28, 512)
        self.fc2=nn.Linear(512, num_classes)

        self.dropout=nn.Dropout(0.5)

    
    def forward(self, x):
        x=F.relu(self.conv1(x))
        x=F.max_pool2d(x, 2)

        x=F.relu(self.conv2(x))
        x=F.max_pool2d(x, 2)

        x=F.relu(self.conv3(x))
        x=F.max_pool2d(x, 2)

        x=x.view(x.size(0), -1)

        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)

        return x


# Defining the number of classes
num_classes = len(os.listdir(os.path.join('Dataset', 'train')))
    
#@ Instantiate the model:
model=FaceRecognitionCNN(num_classes=num_classes).to(device)

#@ Loss function and Optimization:
criterion=nn.CrossEntropyLoss()
optimizer=Adam(model.parameters(), lr=0.001)

#@ Training:
def train():
    num_epochs=10
    for epoch in range(num_epochs):
        model.train() #handles dropout and forward 
        train_loss=0.0
        train_correct=0
        total_train=0

        for images, labels in train_dl:
            images, labels =images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)

            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted=torch.max(outputs, 1)
            train_correct+=(predicted == labels).sum().item()
            total_train+=labels.size(0)

        train_accuracy=train_correct / total_train

        # Validation
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in test_dl:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_accuracy = val_correct / total_val

        print(f'Epoch {epoch+1}/{num_epochs}, '
            f'Train Loss: {train_loss / len(train_dl):.4f}, Train Accuracy: {train_accuracy:.4f}, '
            f'Val Loss: {val_loss / len(test_dl):.4f}, Val Accuracy: {val_accuracy:.4f}')


#@ Saving model:
torch.save(model.state_dict(), 'face_recognition_model.pth')
