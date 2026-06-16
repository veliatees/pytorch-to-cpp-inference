import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))  #parent directory is now = Edge AI
import pandas as pd
import torch
from torch import nn
import numpy as np
import dataset.titanic as titanic

torch.manual_seed(34)

data= titanic.df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
data.dropna(subset=['Embarked'], inplace=True)  #drop rows with missing 'Embarked' values 

print(len(data))

mean_age= data['Age'].mean()
number_of_train= 0.8

data['Age'] = data['Age'].fillna(mean_age)

# print(data['Sex'].unique().sum())
# print(data['Embarked'].unique().sum())

# #ENCODING FOR "SEX", "EMBARKED"
sex_dict= {"male": 0, "female": 1} 
embarked_dict= {"S": 0, "C": 1, "Q":2}

data['Sex'] = data['Sex'].map(sex_dict)
data['Embarked'] = data['Embarked'].map(embarked_dict)

data= data.sample(frac=1, random_state=42)  #shuffle the data

train_data= data.iloc[: int(number_of_train*len(data)) , :]
test_data= data.iloc[int(number_of_train*len(data)):, :]

# print(train_data.head())
# print(test_data.head())

train_X= train_data.drop(columns=['Survived'])
train_y= train_data['Survived']

mean= train_X.mean()
std= train_X.std()
train_X= (train_X - mean) / std


test_X= test_data.drop(columns=['Survived'])
test_y= test_data['Survived']

#NORMALIZE THE TEST WEIGHTS
test_X= (test_X - mean) / std


train_X_tensor= torch.tensor(train_X.values, dtype=torch.float32, requires_grad=False)
train_y_tensor= torch.tensor(train_y.values, dtype=torch.float32, requires_grad=False).unsqueeze(1)  #add an extra dimension to make it a column vector

test_X_tensor= torch.tensor(test_X.values, dtype=torch.float32)
test_y_tensor= torch.tensor(test_y.values, dtype=torch.float32).unsqueeze(1)  #add an extra dimension to make it a column vector    





# print(f"Train X tensor shape: {train_X_tensor.shape}")
# print(f"Train y tensor shape: {train_y_tensor.shape}")
# print(f"Test X tensor shape: {test_X_tensor.shape}")
# print(f"Test y tensor shape: {test_y_tensor.shape}")

#CREATING WEIGHTS AND NORMALIZATION

w1 = torch.randn((7, 12)) / (7 ** 0.5)
w2 = torch.randn((12, 8)) / (12 ** 0.5)
w3 = torch.randn((8, 1)) / (8 ** 0.5)
bias1= torch.zeros((1,), dtype=torch.float32, requires_grad=True)
bias2= torch.zeros((1,), dtype=torch.float32, requires_grad=True)
bias3= torch.zeros((1,), dtype=torch.float32, requires_grad=True)   
w1.requires_grad_(True)
w2.requires_grad_(True)
w3.requires_grad_(True)



epochs= 1200
learning_rate= 0.01


for epoch in range(epochs):
    h1= torch.relu(torch.matmul(train_X_tensor, w1) + bias1)
    h2= torch.relu(torch.matmul(h1, w2) + bias2)
    output= torch.sigmoid(torch.matmul(h2, w3) + bias3)

    loss= nn.BCELoss()(output, train_y_tensor)

    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w3 -= learning_rate * w3.grad
        bias1 -= learning_rate * bias1.grad
        bias2 -= learning_rate * bias2.grad
        bias3 -= learning_rate * bias3.grad

        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()
        bias1.grad.zero_()
        bias2.grad.zero_()
        bias3.grad.zero_()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    elif epoch == epochs - 1:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

with torch.no_grad():
    h1_test= torch.relu(torch.matmul(test_X_tensor, w1) + bias1)
    h2_test= torch.relu(torch.matmul(h1_test, w2) + bias2)
    output_test= torch.sigmoid(torch.matmul(h2_test, w3) + bias3)

print(f"Train Accuracy: {((output > 0.5).float() == train_y_tensor).float().mean().item() * 100:.2f}%")
print(f"Test Accuracy: {((output_test > 0.5).float() == test_y_tensor).float().mean().item() * 100:.2f}%")

print(f"first weight and bias: {w1[0][0].item()}, {bias1.item()}")