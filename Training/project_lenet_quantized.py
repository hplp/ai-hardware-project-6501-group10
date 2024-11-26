# -*- coding: utf-8 -*-
"""Project_LeNet_quantized.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1arT0TzvKCNJzFewQowRLvTldSkiJrLph
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# Define the transformation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Utility functions
def print_size_of_model(model, name="Model"):
    """ Prints the real size of the model """
    torch.save(model.state_dict(), "temp.p")
    print(f'Size of {name} (MB): {os.path.getsize("temp.p") / 1e6}')
    os.remove('temp.p')

def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return correct[:1].view(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size).item()

# Define the LeNet model
class Net(nn.Module):
    def __init__(self, q=False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256, 120, bias=False)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10, bias=False)
        self.q = q
        if q:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.q:
            x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        if self.q:
            x = self.dequant(x)
        return x

# Train function
def train(model, dataloader, cuda=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 100 == 0:
                print(f'Epoch [{epoch + 1}], Step [{i}], Loss: {running_loss / (i + 1):.4f}, Accuracy: {100 * correct / total:.2f}%')

    print('Finished Training')

# Test function
def test(model, dataloader, cuda=False):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model: {accuracy}%')
    return accuracy

# Train and test FP32 model
net_fp32 = Net(q=False).cuda()
print_size_of_model(net_fp32, "FP32 Model")
train(net_fp32, trainloader, cuda=True)
score_fp32 = test(net_fp32, testloader, cuda=True)

# Save the trained FP32 model
torch.save(net_fp32.state_dict(), "lenet_int32.pth")
print("Trained model saved as lenet_int32.pth")

import torch.nn.functional as F

# Simulate INT4 quantization and dequantization
def int4_quantize(tensor):
    scale = 7 / tensor.abs().max()  # Scale factor to map to INT4 range
    quantized_tensor = (tensor * scale).clamp(-7, 7).round().char()  # Quantize to INT4
    return quantized_tensor, scale

def int4_dequantize(quantized_tensor, scale):
    return quantized_tensor.float() / scale  # Dequantize back to float

# Simulate quantized forward pass for INT4 with input quantization
def quantized_forward_int4_with_input_quant(model, x):
    with torch.no_grad():
        # Quantize the input
        x_q, input_scale = int4_quantize(x)  # Quantize input and get scale

        # Manually quantizing each layer's weights
        conv1_w_q, conv1_scale = int4_quantize(model.conv1.weight.data)
        conv2_w_q, conv2_scale = int4_quantize(model.conv2.weight.data)
        fc1_w_q, fc1_scale = int4_quantize(model.fc1.weight.data)
        fc2_w_q, fc2_scale = int4_quantize(model.fc2.weight.data)
        fc3_w_q, fc3_scale = int4_quantize(model.fc3.weight.data)

        # Forward pass with dequantized weights
        x = F.conv2d(int4_dequantize(x_q, input_scale), int4_dequantize(conv1_w_q, conv1_scale), stride=1, padding=0)
        x = model.relu1(x)
        x = model.pool1(x)

        x_q, input_scale = int4_quantize(x)  # Re-quantize the intermediate activation

        x = F.conv2d(int4_dequantize(x_q, input_scale), int4_dequantize(conv2_w_q, conv2_scale), stride=1, padding=0)
        x = model.relu2(x)
        x = model.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor correctly

        # Dequantize and apply the fully connected layers
        x_q, input_scale = int4_quantize(x)  # Quantize the flattened output before fully connected layers

        x = F.linear(int4_dequantize(x_q, input_scale), int4_dequantize(fc1_w_q, fc1_scale))
        x = model.relu3(x)

        x_q, input_scale = int4_quantize(x)  # Re-quantize before the next layer

        x = F.linear(int4_dequantize(x_q, input_scale), int4_dequantize(fc2_w_q, fc2_scale))
        x = model.relu4(x)

        x_q, input_scale = int4_quantize(x)  # Re-quantize before the final layer

        x = F.linear(int4_dequantize(x_q, input_scale), int4_dequantize(fc3_w_q, fc3_scale))

    return x

# Quantized model testing function for INT4 with input quantization
def test_quantized_int4_with_input_quant(model, dataloader, cuda=False):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = quantized_forward_int4_with_input_quant(model, inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the quantized model (INT4) with input quantization on the test images: {accuracy}%')
    return accuracy

# Now test the INT4 quantized model
print("Testing INT4 Quantized Model with Input Quantization...")
score_int4 = test_quantized_int4_with_input_quant(net_fp32, testloader, cuda=True)

# Print model sizes and accuracies for comparison
print(f'FP32 Model Accuracy: {score_fp32}%')
print(f'INT4 Quantized Model Accuracy: {score_int4}%')

# Save the INT4 quantized model
int4_model_state = {
    "conv1_weight": int4_quantize(net_fp32.conv1.weight.data),
    "conv2_weight": int4_quantize(net_fp32.conv2.weight.data),
    "fc1_weight": int4_quantize(net_fp32.fc1.weight.data),
    "fc2_weight": int4_quantize(net_fp32.fc2.weight.data),
    "fc3_weight": int4_quantize(net_fp32.fc3.weight.data),
}
torch.save(int4_model_state, "lenet_int4.pth")
print("Quantized INT4 model saved as lenet_int4.pth")

# INT16 quantization
import torch.nn.functional as F

# Simulate INT16 quantization and dequantization
def int16_quantize(tensor):
    scale = 32767 / tensor.abs().max()  # Scale factor to map to INT16 range
    quantized_tensor = (tensor * scale).clamp(-32767, 32767).round().short()  # Quantize to INT16
    return quantized_tensor, scale

def int16_dequantize(quantized_tensor, scale):
    return quantized_tensor.float() / scale  # Dequantize back to float

# Simulate quantized forward pass for INT16 with input quantization
def quantized_forward_int16_with_input_quant(model, x):
    with torch.no_grad():
        # Quantize the input
        x_q, input_scale = int16_quantize(x)  # Quantize input and get scale

        # Manually quantizing each layer's weights
        conv1_w_q, conv1_scale = int16_quantize(model.conv1.weight.data)
        conv2_w_q, conv2_scale = int16_quantize(model.conv2.weight.data)
        fc1_w_q, fc1_scale = int16_quantize(model.fc1.weight.data)
        fc2_w_q, fc2_scale = int16_quantize(model.fc2.weight.data)
        fc3_w_q, fc3_scale = int16_quantize(model.fc3.weight.data)

        # Forward pass with dequantized weights
        x = F.conv2d(int16_dequantize(x_q, input_scale), int16_dequantize(conv1_w_q, conv1_scale), stride=1, padding=0)
        x = model.relu1(x)
        x = model.pool1(x)

        x_q, input_scale = int16_quantize(x)  # Re-quantize the intermediate activation

        x = F.conv2d(int16_dequantize(x_q, input_scale), int16_dequantize(conv2_w_q, conv2_scale), stride=1, padding=0)
        x = model.relu2(x)
        x = model.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor correctly

        # Dequantize and apply the fully connected layers
        x_q, input_scale = int16_quantize(x)  # Quantize the flattened output before fully connected layers

        x = F.linear(int16_dequantize(x_q, input_scale), int16_dequantize(fc1_w_q, fc1_scale))
        x = model.relu3(x)

        x_q, input_scale = int16_quantize(x)  # Re-quantize before the next layer

        x = F.linear(int16_dequantize(x_q, input_scale), int16_dequantize(fc2_w_q, fc2_scale))
        x = model.relu4(x)

        x_q, input_scale = int16_quantize(x)  # Re-quantize before the final layer

        x = F.linear(int16_dequantize(x_q, input_scale), int16_dequantize(fc3_w_q, fc3_scale))

    return x

# Quantized model testing function for INT16 with input quantization
def test_quantized_int16_with_input_quant(model, dataloader, cuda=False):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = quantized_forward_int16_with_input_quant(model, inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the quantized model (INT16) with input quantization on the test images: {accuracy}%')
    return accuracy

# Now test the INT16 quantized model
print("Testing INT16 Quantized Model with Input Quantization...")
score_int16 = test_quantized_int16_with_input_quant(net_fp32, testloader, cuda=True)

# Print model sizes and accuracies for comparison
print(f'FP32 Model Accuracy: {score_fp32}%')
print(f'INT16 Quantized Model Accuracy: {score_int16}%')

# Save the INT16 quantized model
int16_model_state = {
    "conv1_weight": int16_quantize(net_fp32.conv1.weight.data),
    "conv2_weight": int16_quantize(net_fp32.conv2.weight.data),
    "fc1_weight": int16_quantize(net_fp32.fc1.weight.data),
    "fc2_weight": int16_quantize(net_fp32.fc2.weight.data),
    "fc3_weight": int16_quantize(net_fp32.fc3.weight.data),
}
torch.save(int16_model_state, "lenet_int16.pth")
print("Quantized INT16 model saved as lenet_int16.pth")

# INT8:
import torch.nn.functional as F

# Simulate INT8 quantization and dequantization
def int8_quantize(tensor):
    scale = 127 / tensor.abs().max()  # Scale factor to map to INT8 range
    quantized_tensor = (tensor * scale).clamp(-127, 127).round().char()  # Quantize to INT8
    return quantized_tensor, scale

def int8_dequantize(quantized_tensor, scale):
    return quantized_tensor.float() / scale  # Dequantize back to float

# Simulate quantized forward pass for INT8 with input quantization
def quantized_forward_int8_with_input_quant(model, x):
    with torch.no_grad():
        # Quantize the input
        x_q, input_scale = int8_quantize(x)  # Quantize input and get scale

        # Manually quantizing each layer's weights
        conv1_w_q, conv1_scale = int8_quantize(model.conv1.weight.data)
        conv2_w_q, conv2_scale = int8_quantize(model.conv2.weight.data)
        fc1_w_q, fc1_scale = int8_quantize(model.fc1.weight.data)
        fc2_w_q, fc2_scale = int8_quantize(model.fc2.weight.data)
        fc3_w_q, fc3_scale = int8_quantize(model.fc3.weight.data)

        # Forward pass with dequantized weights
        x = F.conv2d(int8_dequantize(x_q, input_scale), int8_dequantize(conv1_w_q, conv1_scale), stride=1, padding=0)
        x = model.relu1(x)
        x = model.pool1(x)

        x_q, input_scale = int8_quantize(x)  # Re-quantize the intermediate activation

        x = F.conv2d(int8_dequantize(x_q, input_scale), int8_dequantize(conv2_w_q, conv2_scale), stride=1, padding=0)
        x = model.relu2(x)
        x = model.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor correctly

        # Dequantize and apply the fully connected layers
        x_q, input_scale = int8_quantize(x)  # Quantize the flattened output before fully connected layers

        x = F.linear(int8_dequantize(x_q, input_scale), int8_dequantize(fc1_w_q, fc1_scale))
        x = model.relu3(x)

        x_q, input_scale = int8_quantize(x)  # Re-quantize before the next layer

        x = F.linear(int8_dequantize(x_q, input_scale), int8_dequantize(fc2_w_q, fc2_scale))
        x = model.relu4(x)

        x_q, input_scale = int8_quantize(x)  # Re-quantize before the final layer

        x = F.linear(int8_dequantize(x_q, input_scale), int8_dequantize(fc3_w_q, fc3_scale))

    return x

# Quantized model testing function for INT8 with input quantization
def test_quantized_int8_with_input_quant(model, dataloader, cuda=False):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = quantized_forward_int8_with_input_quant(model, inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the quantized model (INT8) with input quantization on the test images: {accuracy}%')
    return accuracy

# Now test the INT8 quantized model
print("Testing INT8 Quantized Model with Input Quantization...")
score_int8 = test_quantized_int8_with_input_quant(net_fp32, testloader, cuda=True)

# Print model sizes and accuracies for comparison
print(f'FP32 Model Accuracy: {score_fp32}%')
print(f'INT8 Quantized Model Accuracy: {score_int8}%')

# Save the INT16 quantized model
int8_model_state = {
    "conv1_weight": int8_quantize(net_fp32.conv1.weight.data),
    "conv2_weight": int8_quantize(net_fp32.conv2.weight.data),
    "fc1_weight": int8_quantize(net_fp32.fc1.weight.data),
    "fc2_weight": int8_quantize(net_fp32.fc2.weight.data),
    "fc3_weight": int8_quantize(net_fp32.fc3.weight.data),
}
torch.save(int8_model_state, "lenet_int8.pth")
print("Quantized INT8 model saved as lenet_int8.pth")

# Print model sizes and accuracies for comparison
print(f'FP32 Model Accuracy: {score_fp32}%')
print(f'INT16 Quantized Model Accuracy: {score_int16}%')
print(f'INT8 Quantized Model Accuracy: {score_int8}%')
print(f'INT4 Quantized Model Accuracy: {score_int4}%')