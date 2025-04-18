import torch
from preprocess import X_train, y_train
from ann import ANN

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

model = ANN().to("cpu")

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 32
num_epochs = 100
train_loader = torch.utils.data.DataLoader(dataset=CustomDataset(X_train, y_train), batch_size=batch_size)

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    total = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = torch.round(output.data)
        total += batch_size
        correct += (predicted == target).sum().item()

    # Print epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# Save the model
torch.save(model.state_dict(), 'ann.pt')
