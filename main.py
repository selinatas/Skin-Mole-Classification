import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

from dataset import train_loader, valid_loader, test_loader, idx_to_class, class_to_idx

test_loss_resnet, test_acc_resnet = None, None
test_loss_densenet, test_acc_densenet = None, None
test_loss_customcnn, test_acc_customcnn = None, None

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for img, labels in loader:
            img, labels = img.to(device), labels.to(device)
            output = model(img)
            loss = criterion(output, labels)
            total_loss += loss.item() * img.size(0)
            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == labels).item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

#######################################################
#      RESNET-50
#######################################################

resnet = models.resnet50(pretrained=True)

# to freeze the parameters in conv layers
for param in resnet.parameters():
    param.requires_grad = False

num_features_res = resnet.fc.in_features
num_classes = len(class_to_idx)
resnet.fc = nn.Linear(num_features_res, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet.to(device)

num_epoch = 10

for epoch in range(num_epoch):
    resnet.train()
    run_loss = 0
    for img, labels in train_loader:
        img, labels = img.to(device), labels.to(device)
        optimizer.zero_grad() 
        output = resnet(img)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * img.size(0)
    epoch_loss = run_loss / len(train_loader.dataset)
    print(f"ResNet-50 Epoch {epoch+1}/{num_epoch}, Loss: {epoch_loss:.4f}")

    resnet.eval()
    val_loss, val_acc = evaluate_model(resnet, valid_loader, criterion, device)
    print(f"ResNet-50 Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

test_loss_resnet, test_acc_resnet = evaluate_model(resnet, test_loader, criterion, device)
print(f"ResNet-50 Test Loss: {test_loss_resnet:.4f}, Accuracy: {test_acc_resnet:.4f}")

#######################################################
#      DENSENET-121
#######################################################

densenet = models.densenet121(pretrained=True)

for param in densenet.parameters():
    param.requires_grad = False

num_features_densenet = densenet.classifier.in_features
densenet.classifier = nn.Linear(num_features_densenet, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(densenet.classifier.parameters(), lr=0.001)
densenet.to(device)

for epoch in range(num_epoch):
    densenet.train()
    run_loss = 0
    for img, labels in train_loader:
        img, labels = img.to(device), labels.to(device)
        optimizer.zero_grad()
        output = densenet(img)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * img.size(0)
    epoch_loss = run_loss / len(train_loader.dataset)
    print(f"DenseNet-121 Epoch {epoch+1}/{num_epoch}, Loss: {epoch_loss:.4f}")

    densenet.eval()
    val_loss, val_acc = evaluate_model(densenet, valid_loader, criterion, device)
    print(f"DenseNet-121 Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

test_loss_densenet, test_acc_densenet = evaluate_model(densenet, test_loader, criterion, device)
print(f"DenseNet-121 Test Loss: {test_loss_densenet:.4f}, Accuracy: {test_acc_densenet:.4f}")

#######################################################
#      CUSTOM CNN
#######################################################

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(512 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(-1, 512 * 16 * 16) 
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CustomCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device)

for epoch in range(num_epoch):
    model.train()
    run_loss = 0
    for img, labels in train_loader:
        img, labels = img.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * img.size(0)
    epoch_loss = run_loss / len(train_loader.dataset)
    print(f"Custom CNN Epoch {epoch+1}/{num_epoch}, Loss: {epoch_loss:.4f}")

    model.eval()
    val_loss, val_acc = evaluate_model(model, valid_loader, criterion, device)
    print(f"Custom CNN Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

test_loss_customcnn, test_acc_customcnn = evaluate_model(model, test_loader, criterion, device)
print(f"Custom CNN Test Loss: {test_loss_customcnn:.4f}, Accuracy: {test_acc_customcnn:.4f}")

#######################################################
#      RESULTS TABLE
#######################################################


model_results = {
    'Model': ['ResNet-50', 'DenseNet-121', 'Custom CNN'],
    'Test Loss': [round(test_loss_resnet,4), round(test_loss_densenet,4), round(test_loss_customcnn,4)],
    'Test Accuracy': [round(test_acc_resnet,4), round(test_acc_densenet,4), round(test_acc_customcnn,4)]
}

results_df = pd.DataFrame(model_results)

print(results_df)

# Save the DataFrame as a PNG image
fig, ax = plt.subplots(figsize=(8, 2)) 
ax.axis('tight')
ax.axis('off')
tbl = table(ax, results_df, loc='center', cellLoc='center', colWidths=[0.2]*len(results_df.columns))
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.2, 1.2)

# Save the table as an image
table_image_path = "model_test_results.png"
plt.savefig(table_image_path)

print(f"The results table has been saved to {table_image_path}")


#######################################################
#      HELPER FUNCTION FOR STREAMLIT
#######################################################

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(model, image, device):
    model.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0).to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_name = idx_to_class[predicted.item()]
    return class_name