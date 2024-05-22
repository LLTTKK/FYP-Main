import torch
import torch.nn as nn
from cnn import ConvNeuralNet
from PIL import Image
import numpy as np


class BinaryClassifierOfFullOrNot(nn.Module):
    def __init__(self, input_feature_size):
        self.fc_layer_1 = nn.Linear(input_feature_size, 1024)
        self.fc_layer_2 = nn.Linear(1024, 512)
        self.fc_layer_3 = nn.Linear(512, 128)
        self.fc_layer_4 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc_layer_1(x)
        x = nn.functional.relu(x)
        x = self.fc_layer_2(x)
        x = nn.functional.relu(x)
        x = self.fc_layer_3(x)
        x = nn.functional.relu(x)
        x = self.fc_layer_4(x)


img = Image.open("C:\\Users\\ouhkstaff\\Desktop\\test_img.jpg")

img_tensor = np.array(img)

x = torch.tensor(img_tensor)

temp_x = torch.ones(size=(1, 3, 1300, 866))

print(x.shape)
print(temp_x.shape)

backbone_model = ConvNeuralNet(2)

output = model(x)


