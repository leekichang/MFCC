# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 1. File Loading

# %%
import torch
import pandas as pd
import numpy as np
import os

# load data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# train
from torch import nn
from torch.nn import functional as F
import torch.nn.init

# visualization
import matplotlib.pyplot as plt


# %%
batch_size = 32
num_workers = 40
train_set = 'slim_train_set2.csv'
test_set = 'slim_test_set2.csv'


# %%
class MFCCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.raw = pd.read_csv(csv_file, header=None)
        self.label = torch.IntTensor(np.array(self.raw[0].values).reshape(len(self.raw[0].values), 1))
        self.len = len(self.label)
        self.data = torch.Tensor(np.array(self.raw.loc[:,1:]).reshape(len(self.label),100,500))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

print("Loading data...")

dataset = MFCCDataset(train_set)

dataloader = torch.utils.data.DataLoader(
   dataset, batch_size = batch_size, num_workers = num_workers, drop_last=True
)
print("data is ready!")

# %% [markdown]
# # 2. Neural Network

# %%
import random
USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용\
print("다음 기기로 학습합니다:", device)
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


# %%
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1) -> 1, 100, 1000, 1
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=13, stride=1, padding=6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=10, stride=10))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=5, stride=5))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(1 * 5 * 128, 3, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out


# %%
# CNN 모델 정의
learning_rate = 0.001
training_epochs = 50
model = CNN()   #.to(device)
model = torch.nn.DataParallel(model)
model.cuda()
criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(dataloader)
print('총 배치의 수 : {}'.format(total_batch))


# %%
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in dataloader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.reshape(32,1,100,500).to(device)
        Y = Y.reshape(32,1)[:,0].to(device, dtype=torch.int64)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


# %%
print("Loading testset...")

testset = MFCCDataset(test_set)

dataloader = torch.utils.data.DataLoader(
   testset, batch_size = batch_size, num_workers = num_workers, drop_last=True
)
print("testset is ready!")


# %%
with torch.no_grad():
    X_test = testset.data.view(len(testset), 1, 100, 500).float().to(device)
    Y_test = testset.label.to(device, dtype=torch.int64)[:,0]

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())


