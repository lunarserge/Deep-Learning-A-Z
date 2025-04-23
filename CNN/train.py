import random
import torch
from torchvision.transforms import v2
from cnn import CNN, transforms_base

class RandomZoom:
    def __call__(self, img):
        w, h = img.size
        zoom = random.uniform(0.8, 1.2)
        img = v2.Resize((int(h*zoom), int(w*zoom)))(img)
        return v2.CenterCrop((h, w))(img)

transforms_aug = v2.Compose([
    v2.RandomAffine(0, shear=(-11.31,11.31,-11.31,11.31)),
    RandomZoom(),
    v2.RandomHorizontalFlip(),
    transforms_base
])

from torchvision import datasets
train_dataset = datasets.ImageFolder('dataset/training_set', transform=transforms_aug)
test_dataset = datasets.ImageFolder('dataset/test_set', transform=transforms_base)

from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset)

model = CNN().to("cpu")
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 32
num_epochs = 25

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data).reshape(batch_size)
        target = target.float()
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = torch.round(output.data)
        correct += (predicted == target).sum().item()

    correct_test = 0

    for data, target in test_loader:
        output = model(data)
        target = target.float()

        predicted = torch.round(output.data)
        correct_test += (predicted == target).item()

    # Print epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / (len(train_loader)*batch_size)
    epoch_acc_test = 100 * correct_test / len(test_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy (train): {epoch_acc:.2f}%, Accuracy (test): {epoch_acc_test:.2f}%")

# Save the model
torch.save(model.state_dict(), 'cnn.pt')
