import torch
import torch.nn as nn
from cnn import ConvNeuralNet
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor


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
        return x



img = Image.open("./test_img.jpg")

img_tensor = np.array(img)

x = ToTensor()(img_tensor)

x = x.unsqueeze(0)

print("Image shape: ", x.shape)

backbone_model = ConvNeuralNet(2)

output = backbone_model(x)

print("Output shape from backbone: ", output.shape)

feature_size = int(output.shape[1])

classifier_model = BinaryClassifierOfFullOrNot(feature_size)

final_output = classifier_model(output)

print(final_output)

