IMAGE = ''

import torch
from cnn import CNN, transforms_base

# Load the model
model = CNN()
model.load_state_dict(torch.load('cnn.pt'))
model.eval()

from torchvision.io import read_image
img = transforms_base(read_image(IMAGE))
img = img.unsqueeze(0) # Add batch dimension

# Predict
with torch.no_grad():
    output = model(img)

print(output)
