import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))  #parent directory is now = Edge AI
import pandas as pd
import torch
from torch import nn
import numpy as np
import dataset.titanic as titanic

data= titanic.df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
data.dropna(subset=['Embarked'], inplace=True)  #drop rows with missing 'Embarked' values 

print(len(data))

sum= 0
for value in data['Age']:
    if pd.isna(value):
        continue
    else:
        sum+= value
        mean_age= round(sum/len(data['Age']), 1)
    
print(f"Mean age: {mean_age}")
number_of_train= 0.8

data['Age'] = data['Age'].fillna(mean_age)


train_data= data.iloc[: int(number_of_train*len(data)) , :]
test_data= data.iloc[int(number_of_train*len(data)):, :]

print(f"Length of training data: {len(train_data)}")
print(f"Length of test data: {len(test_data)}")

# print(train_data.to_numpy().dtype)
# print(test_data.to_numpy().dtype)

#print(train_data.head())
#print(test_data.head())



# #ENCODING FOR SEX, CABIN, EMBARKED
# sex_dict= {"Male": 0, "Female": 1} 
# cabin_dict= {}
# embarked_dict= {"S": 0, "C": 1, "Q":2, "nan": 3}