
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import glob
import torch.nn as nn
from transformers import  CvtForImageClassification
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
class CustomDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.loc[idx, 'image_path']
        img_path = os.path.join(r'E:\datasets\artifact', img_path)
        #print(img_path)
        #print("aaaa####")
        image = Image.open(img_path).convert("RGB")
        self.data.loc[idx, 'target']
        print(self.data.loc[idx, 'target'])
        label = int(self.data.loc[idx, 'target'])
        if self.transform:
            image = self.transform(image)

        return image, label
def compute_class_weights(n_samples_class0, n_samples_class1):
    total = n_samples_class0 + n_samples_class1
    weight_class0 = total / (2 * n_samples_class0)
    weight_class1 = total / (2 * n_samples_class1)
    return weight_class0, weight_class1
class CustomClassifier(nn.Module):
    def __init__(self):
        super(CustomClassifier, self).__init__()
        # First Hidden Layer
        self.fc1 = nn.Linear(384, 256)
        self.mish1 = nn.Mish(inplace=False)
        self.norm1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=0.5)
        # Second Hidden Layer
        self.fc2 = nn.Linear(256, 128)
        self.mish2 = nn.Mish(inplace=False)
        self.norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.3)
        # Output Layer
        self.fc_out = nn.Linear(128, 2)
    def forward(self, x):
        x = self.dropout1(self.norm1(self.mish1(self.fc1(x))))
        x = self.dropout2(self.norm2(self.mish2(self.fc2(x))))
        x = self.fc_out(x)
        return x


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_data = CustomDataset(r"E:\datasets\artifact\train.csv", transform=transform)
    test_data = CustomDataset(r"E:\datasets\artifact\test.csv", transform=transform)
    n_samples_class0 = sum(train_data.data["target"] == 0.0)
    n_samples_class1 = sum(train_data.data["target"] == 1.0)
    weight_class0, weight_class1 = compute_class_weights(n_samples_class0, n_samples_class1)
    samples_weights = [weight_class0 if label == 0.0 else weight_class1 for label in train_data.data["target"]]
    weighted_sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
    train_loader = DataLoader(train_data, batch_size=128, sampler=weighted_sampler, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=6, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([weight_class0, weight_class1]).to(device))
    model = CvtForImageClassification.from_pretrained('microsoft/cvt-13')
    model = model.to(device)
    model.classifier = CustomClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-8)
    scaler = GradScaler()
    list_of_files = glob.glob('models/model_epoch_*.pth')
    if list_of_files:  # Check if the list is not empty
        latest_file = max(list_of_files, key=os.path.getctime)
        checkpoint = torch.load(latest_file)
        train_losses = checkpoint['train_losses']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
        avg_loss_loaded = checkpoint.get('avg_loss', None)
        model.train()
        total_epochs = 50
    else:
        total_epochs = 50
        starting_epoch = 0
        avg_loss_loaded = None
        train_losses = []

    for g in optimizer.param_groups:
        print(g['lr'])

    for epoch in range(starting_epoch, total_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}, Learning Rate: {current_lr:.6f}")

        model.train()
        total_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}", unit="batch") as progress_bar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.logits, labels)

                # Gradient clipping
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                progress_bar.set_postfix({"Loss": loss.item()})
                progress_bar.update()

                # Free up memory
                del inputs, labels, outputs

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_loss': avg_train_loss,
            'scaler_state_dict': scaler.state_dict(),
            'train_losses': train_losses,
            # change path for your own
        }, f"weights/model_epoch_{epoch}.pth")

        torch.cuda.empty_cache()
    model.eval()
    test_loss = 0
    all_predicted = []
    all_labels = []
    with torch.no_grad():  # Disables gradient calculation for evaluation, which reduces memory usage
        with tqdm(total=len(test_loader), desc="Evaluation", unit="batch") as progress_bar:
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.logits, labels)

                test_loss += loss.item()
                _, predicted = outputs.logits.max(1)
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix({"Test Loss": loss.item()})
                progress_bar.update()

    all_labels = np.array(all_labels)
    all_predicted = np.array(all_predicted)

    avg_test_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_predicted)
    precision = precision_score(all_labels, all_predicted, average='macro')
    recall = recall_score(all_labels, all_predicted, average='macro')
    f1 = f1_score(all_labels, all_predicted, average='macro')

    print(f'Average Test Loss: {avg_test_loss:.4f}')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    cm = confusion_matrix(all_labels, all_predicted)
    # Plot
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    show_absolute=True,
                                    show_normed=False)
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig("confusion_matrix.png")

    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()
    plt.savefig("loss.png")










