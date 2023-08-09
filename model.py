import os
import csv
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = list(csv.reader(open('./data/Data.csv','r')))
user_pick_list = data[0]

train_x = []
train_y = []
# test_x = []
# test_y = []
for row in data[1:]:
    train_x.append(list(map(int,row[1:-4]))) #data
    train_y.append(list(map(int, row[-4:-3]))) #answer

# for row in data[12:]:
#     print(row)
#     test_x.append(list(map(int,row[1:-4]))) #data
#     test_y.append(list(map(int,row[-4:-3]))) #answer

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(40, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
# Dataset 상속
class CustomDataset(Dataset): 
  def __init__(self,x,y):
    self.x_data = x
    self.y_data = y

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y



training_data = CustomDataset(train_x,train_y)

# test_data = CustomDataset(test_x,test_y)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        y = y.type(torch.LongTensor)
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y.squeeze(dim=-1))

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch + 1) * len(X) == size:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    print(size, num_batches)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            y = y.type(torch.LongTensor)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.squeeze(dim=-1)).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()/len(X)
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

train_dataloader = DataLoader(training_data, batch_size=5, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=5, shuffle=True)
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    # test(test_dataloader, model, loss_fn)
torch.save(model, 'model.pth')
print("Done!")
 
print(model)