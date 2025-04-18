import torch
from preprocess import X_test, y_test
from ann import ANN

# Load the model
model = ANN()
model.load_state_dict(torch.load('ann.pt'))
model.eval()

# Predict
with torch.no_grad():
    output = torch.round(model(torch.tensor(X_test))).numpy()

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, output)
print(cm)
print(accuracy_score(y_test, output))
