from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import random
from timeit import default_timer as timer
from pathlib import Path
from typing import Tuple, Dict, List
import torch.nn.functional as f

device = "cuda" if torch.cuda.is_available() else "cpu"
device

class customdataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, drop):
        df = pd.read_csv(csv_path, header=None)
        df = df.values
        df = df[drop:, :]
        self.labels = df[:, -1]
        self.datas = df[:, :-1]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = torch.LongTensor([self.labels[idx]])
        data = torch.Tensor(self.datas[idx]).unsqueeze(0)
        return data, label
    
mitbih_test = customdataset('C:/Users/Asus/.vscode/VSC/mlmed2024/Heartbeat Data/mitbih_test.csv', 6700)
mitbih_train = customdataset('C:/Users/Asus/.vscode/VSC/mlmed2024/Heartbeat Data/mitbih_train.csv', 16500)

batch_size = 32

mitbih_train_dataloader = DataLoader(mitbih_train, batch_size=32, shuffle=True)
mitbih_test_dataloader = DataLoader(mitbih_test, batch_size=32, shuffle=False)

print(f"Dataloaders: {mitbih_train_dataloader, mitbih_test_dataloader}")
print(f"Length of train dataloader: {len(mitbih_train_dataloader)} batches of {batch_size}")
print(f"Length of test dataloader: {len(mitbih_test_dataloader)} batches of {batch_size}")

class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv1d(32, 32, 5, padding='same')
        self.conv2 = nn.Conv1d(32, 32, 5, padding='same')
        self.pool = nn.MaxPool1d(5, 2)
        
    def forward(self, x):
        output = f.relu(self.conv1(x))
        output = self.conv2(output)
        output = f.relu(output+x)
        output = self.pool(output)
        return output
    
class Model_mitbih(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv1d(1, 32, 5, padding='same') 
        self.res = []
        for i in range(5):
            self.res.append(ResBlock().to(device))
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 5)
        
    def forward(self, x):
        output = self.conv(x)
        for i in range(5):
            output = self.res[i](output)
        output = nn.Flatten()(output)
        output = f.relu(self.fc1(output))
        output = f.softmax(self.fc2(output))
        return output
    
model_1 = Model_mitbih().to(device)
next(model_1.parameters()).device

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

def accuracy_fn(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_step(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module,
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
               accuracy_fn, device: torch.device = device):
    model.to(device)
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device).squeeze(1)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        

    train_loss = train_loss / len(data_loader)
    train_acc = train_acc / len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    return train_loss, train_acc

def test_step(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn, device: torch.device = device):
    model.to(device)
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device).squeeze(1)

            test_pred = model(X)

            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
            
            test_pred_label = test_pred.argmax(dim=1)
            
        # Adjust metrics and print out
        test_loss = test_loss / len(data_loader)
        test_acc = test_acc / len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
    return test_loss, test_acc

def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,loss_fn: torch.nn.Module = nn.CrossEntropyLoss()):
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
        }
    for epoch in range (6): #tqdm(range(epochs)):
        
        train_loss, train_acc = train_step(data_loader = mitbih_train_dataloader,
                                           model = model_1, loss_fn = loss_fn, optimizer = optimizer, accuracy_fn = accuracy_fn, device = device)
        test_loss, test_acc = test_step(data_loader = mitbih_test_dataloader,
                                        model = model_1, loss_fn = loss_fn, accuracy_fn = accuracy_fn, device = device)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

def print_train_time(start: float, end: float, device: torch.device = None):
  total_time = end - start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  return total_time

from timeit import default_timer as timer


train_time_start_model_1 = timer()

model_1_results = train(model = model_1,
                        train_loader = mitbih_train_dataloader,
                        test_loader = mitbih_test_dataloader,
                        optimizer = optimizer,
                        loss_fn = loss_fn)

train_time_end_model_1 = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_model_1, end=train_time_end_model_1, device=device)

def eval_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn, device: torch.device = device):
  loss, acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device).squeeze(1)
      y_pred = model(X)

      loss += loss_fn(y_pred, y)
      acc += accuracy_fn(y_true = y, y_pred = y_pred.argmax(dim = 1))

    loss /= len(data_loader)
    acc /= len(data_loader)

  return {"model_name": model.__class__.__name__, 
          "model_loss": loss.item(),
          "model_acc": acc}
  
eval_model(model = model_1, data_loader = mitbih_test_dataloader, loss_fn = loss_fn, accuracy_fn = accuracy_fn, device = device)

model_1_results.keys()

def plot_loss_curves(results: Dict[str, List[float]]):
    loss = torch.tensor(results['train_loss'])
    test_loss = torch.tensor(results['test_loss'])

    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    epochs = range(len(results['train_loss']))
 
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
plot_loss_curves(model_1_results)

def precision_recall_f1(y_true: np.array, y_pred: np.array, average: str = "weighted"):
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
    return {"precision": precision, "recall": recall, "f1": f1}

def ROC_AUC( y_true: np.array, y_pred: np.array):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)

def confusion_matrix(y_true: np.array, y_pred: np.array):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred)