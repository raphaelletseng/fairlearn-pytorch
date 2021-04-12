#MLP for REGRESSION (multilayer perceptron model)

import numpy as np
import pandas as pd
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_

class CSVDataset(Dataset):
    def __init__(self):
        path = 'adult.csv'
        sensitive = 'sex'
        #one hot encoding
        cols = ['age', 'workclass', 'fnlwgt', 'education',
                    'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain',
                    'capital-loss', 'hours-per-week', 'native-country', 'y']

        df = read_csv(path, names = cols, nrows = 100)
        df = df.replace({'?': np.nan})
        df['y'] = df['y'].apply(lambda x:0 if ">50K" in x else 1)
        df = df.dropna()
        df = df.sample(frac=1).reset_index(drop=True)

        print(df.shape)

    #    A = df[sensitive]
    #    X = df.drop(labels = [sensitive], axis = 1)
    #    X = df
        #df = pd.get_dummies(df)
        print(df.shape)
        self.X = df.astype('float32').to_numpy()
    #    self.X = self.X.to_frame().T
        print("X encoded\n")
        #self.X = X
        print("X type: ")
        print(type(self.X))
        print("X len")
        print(len(self.X))
    #    print("X SHAPE: ")
    #    print(self.X.shape())
        #print(self.X.iloc[5])

        self.y = df['y'].astype('float32')

        self.y = self.y.to_numpy()
        print("Y type: ")
        print(type(self.y))
        print(self.y)

        self.y = self.y.reshape((len(self.y), 1))
        print("Y LEN")
        print(len(self.y))

    #    self.X = df.values[:,  :-1].astype('float32')
    #    self.y = df.values[:, -1].astype('float32')
    #    self.y = self.y.reshape((len(self.y),1))
    def X (self):
        return self.X

    def y (self):
        return self.y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        print("Getting item: %d" %(idx))
        #print(type(self.X.iloc[idx]))
        #return [self.X[idx], self.y[idx]]
        return [self.X[idx], self.y[idx]]

    def get_splits(self, n_test=0.2):
        test_size= round(n_test * len(self.X))
        train_size= len(self.X) - test_size
        return  random_split(self, [train_size, test_size])

'''
class LoadDataset(Dataset):
    def __init__(self, data, mode, sensitive_col):
        self.len = data.shape[0]
        print(data.head())
        categorical_columns = ['workclass', 'education', 'marital-status',
                                   'occupation', 'relationship', 'race',
                                   'sex', 'native-country']

        numerical_columns = ['education-num', 'capital-gain',
                                 'capital-loss', 'hours-per-week']

        for c in categorical_columns:
            data[c] = data[c].astype('category')
'''

def prep_data():
    dataset = CSVDataset()
    train, test = dataset.get_splits()
    #print(dataset[5])
    train_dl = DataLoader(train, batch_size=32, shuffle = True)
    test_dl = DataLoader(test, batch_size=1024, shuffle = False)
    return train_dl, test_dl
    #return train, test

train_dl, test_dl = prep_data()
print("len train, len test: ")
print(len(train_dl.dataset), len(test_dl.dataset))
print("TYPE train_dl")
print(type(train_dl))
#print("SHAPE")
#print(train_dl.shape())


class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()

        self.hidden1 = Linear(n_inputs, 10)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()

        self.hidden2 = Linear(10, 8)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()

        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)

    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)

        X = self.hidden2(X)
        X = self.act2(X)

        X = self.hidden3(X)
        return X

def train_model(train_dl, model):
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum = 0.9)
    for epoch in range(100):
        for i, (inputs, targets) in enumerate(train_dl):
            print(i)
            print("\n")
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()

def eval_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        #actual = actual.reshape((len(actual)), 1)

        predictions.append(yhat)
        actuals.append(yhat)
    predictions, actuals = vstack(predictions), vstack(actuals)
    mse = mean_squared_error(actuals, predictions)
    return mse

def predict(row, model):
    row = Tensor([row])
    yhat = model(row)
    yhat = yhat.detach().numpy()
    return yhat


model = MLP(243)
#print("Train model.\n")
#train_model(train_dl, model)
#print("Model trained!\n")

#mse = eval_model(test_dl, model)
#if(mse == 0): print("this seems wrong bud\n")
#print('MSE: %.3f, RMSE: %.3f' %(mse, sqrt(mse)))

#row = train_dl.dataset[3]
#print(row[0])
#print(len(row[0]))

#yhat = predict(row[0], model)
#print("Pred: %.3f" %yhat)
