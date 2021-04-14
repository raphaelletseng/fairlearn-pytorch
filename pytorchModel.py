#MLP for REGRESSION (multilayer perceptron model)

import numpy as np
import pandas as pd
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor

from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn import BCELoss
from torch.nn.init import xavier_uniform_

from fairlearn.reductions import GridSearch
from fairlearn.reductions import DemographicParity, ErrorRate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from opacus import PrivacyEngine

import fairlearn.metrics as flm
import sklearn.metrics as skm
from fairlearn.metrics import true_positive_rate
from fairlearn.metrics import MetricFrame
from sklearn.metrics import confusion_matrix
import wandb
wandb.login()
run_name = "run4-sex-dp"
noise = 1.0
enable_dp = True

class CSVDataset(Dataset):
    def __init__(self):
        path = 'adult.csv'
        sensitive = 'sex'
        #one hot encoding
        cols = ['age', 'workclass', 'fnlwgt', 'education',
                    'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain',
                    'capital-loss', 'hours-per-week', 'native-country', 'y']

        df = read_csv(path, names = cols, nrows = 2000)
        df = df.replace({'?': np.nan})
        df['y'] = df['y'].apply(lambda x:0 if ">50K" in x else 1)
        df = df.dropna()
        df = df.sample(frac=1).reset_index(drop=True)

        y = df['y']
        df = df.drop(labels = ['y'], axis =1)

        A = df['sex']
        X = df.drop(labels = ['sex'], axis = 1)
        self.A = A#.to_numpy() #Series (make into numpy array)

        print(X.shape)
        X = pd.get_dummies(X)
        self.raw_X = X
        print(X.shape)
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns = X.columns)
        self.X = X_scaled
        print(f"Type of X: {type(self.X)} \n") #DataFrame

        le = LabelEncoder()

        #self.y = self.y.to_numpy()
        y = le.fit_transform(y)
        print("Y type: ") #numpy.ndArray
        print(type(self.y))

        self.y = y
    def raw_X(self):
        return self.raw_X

    def A (self):
        return self.A

    def X (self):
        return self.X

    def y (self):
        return self.y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        print("Getting item: %d" %(idx))
        return [self.X.iloc[idx], self.y[idx]]#, self.A[idx]]

    def get_splits(self, n_test=0.2):
        test_size= round(n_test * len(self.X))
        train_size= len(self.X) - test_size
        return  random_split(self, [train_size, test_size])

wandb.init(project="fairlearn-pytorch", name =run_name, config={
    "run_name": run_name,
    "architecture": "MLP",
    "dataset": "adult",
    "batch_size": 32,
    "n_epoch": 20,
    "learning_rate": 0.001,
    "noise": noise
})
config = wandb.config


dataset = CSVDataset()
train, test = dataset.get_splits()
train_dl = DataLoader(train, batch_size=32, shuffle = True)
#test_dl = DataLoader(test, batch_size=1024, shuffle = False)

X = dataset.raw_X

X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(
    dataset.X, dataset.y, dataset.A, test_size = 0.2, random_state = 0, stratify = dataset.y
)

# Does this go in the dataloaded? or in train method?
X_train = X_train.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
X_test = X_test.reset_index(drop = True)
A_test = A_test.reset_index(drop=True)

#def create_model():
#regression??
class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()

        self.hidden1 = Linear(n_inputs, 12)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()

        self.hidden2 = Linear(12, 8)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()

        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)

    def forward(self, X, **kwargs):
        X = self.hidden1(X)
        X = self.act1(X)

        X = self.hidden2(X)
        X = self.act2(X)

        X = self.hidden3(X)
        return X

# Add comments lovely.
class SampleWeightNN(NeuralNetClassifier):
    def __init__(self, *args, criterion__reduce = False, **kwargs):
        super().__init__(*args, criterion__reduce=criterion__reduce, **kwargs)

    def fit(self, X, y, sample_weight = None):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy().astype('float32')
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        if sample_weight is not None and isinstance(sample_weight, (pd.DataFrame, pd.Series)):
            sample_weight = sample_weight.to_numpy()
        y = y.reshape([-1, 1])
        sample_weight = sample_weight if sample_weight is not None else np.ones_like(y)
        X = {'X': X, 'sample_weight': sample_weight}
        return super().fit(X, y)

    def predict(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy().astype('float32')
        return (super().predict_proba(X) > 0.5).astype(np.float)

    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        loss_unreduced = super().get_loss(y_pred, y_true.float(), X, *args, **kwargs)
        sample_weight = X['sample_weight']
        sample_weight = sample_weight.to(loss_unreduced.device).unsqueeze(-1)
        loss_reduced = (sample_weight * loss_unreduced).mean()
        return loss_reduced

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = SampleWeightNN(
    MLP(2123),
    max_epochs = 20,
    optimizer = SGD,
    lr = 0.001,
    batch_size = 32,
    train_split = None,
    iterator_train__shuffle = True,
    criterion = MSELoss,
    device = device
)

print("Training unmitigated")
unmitigated_predictor = net
unmitigated_predictor.fit(X_train, Y_train)
unmitigated_prediction = unmitigated_predictor.predict(X_test)
acc_score_um = skm.accuracy_score(Y_test, unmitigated_prediction)
print(f"Accuracy score um: {acc_score_um}")

optimizer = SGD(net.parameters(), lr= 0.001)

train_size = len(X_train)
#-------------------------------DP--------------------------------#
if(enable_dp):
    if noise > 0:
        privacy_engine = PrivacyEngine(
            net,
            batch_size=32,
            sample_size = train_size,
            alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier = noise,
            max_grad_norm = 1.0,
            secure_rng = False,
        )
        privacy_engine.attach(optimizer)


#--------------------------------- Grid Search -------------------- #

estimator = net
disparity_moment = DemographicParity()
print("Here we go.\n")
sweep = GridSearch(estimator, disparity_moment, grid_size = 71)
sweep.fit(X_train, Y_train, sensitive_features = A_train)
print("sweep fit done.\n")
predictors = sweep.predictors_

print("Going to iterate. \n")

errors, disparities, accuracies  = [], [], []
for m in predictors:
    def classifier(X): return m.predict(X)
    error = ErrorRate()
    #print(error, "\n")
    error.load_data(X_train, pd.Series(Y_train), sensitive_features = A_train)
    disparity = DemographicParity()
    disparity.load_data(X_train, pd.Series(Y_train), sensitive_features= A_train)
    errors.append(error.gamma(classifier)[0])
    disparities.append(disparity.gamma(classifier).max())
    accuracy=skm.accuracy_score(Y_test, classifier(X_test))
    accuracies.append(accuracy)

    error_log = error.gamma(classifier)[0]
    disparity_log = disparity.gamma(classifier).max()

    wandb.log({
                'error': error_log,
                'disparity': disparity_log,
                'acc': accuracy})

print("All results voila: \n")
all_results = pd.DataFrame({"predictor": predictors, "error": errors, "disparity": disparities})
all_results2 = pd.DataFrame({"error": errors, "disparity": disparities, "accuracy": accuracies})

non_dominated = []
for row in all_results.itertuples():
    error_for_lower_or_eq_disparity = all_results["error"][all_results["disparity"] <= row.disparity]
    if row.error <= error_for_lower_or_eq_disparity.min():
        non_dominated.append(row.predictor)
print(all_results2)
print("#################################################")
#print(non_dominated)
file_path = "out//all_results." + run_name
file_object = open(file_path, "a+")
all_results2.to_csv(file_object, index = False)
file_object.close()


"""
result = """
#==============
#Test set: {}

#error: {:.4f}
#disparity: {.4f}
""".format(run_name,
            all_results[1],
            all_results[2]
            )

log_dict = {"error": all_results[0],
            "disparity": all_results[1]
            }
print(log_dict)
wandb.log(log_dict)

"""

"""
from . import package_test_common as ptc
def test_expgrad_classification():
    estimator = create_model()
    disparity_moment = DemographicParity()

    ptc.run_expgrad_classification(estimator, disparity_moment)


def test_gridsearch_classification():
    estimator = create_model()
    disparity_moment = DemographicParity()

    ptc.run_gridsearch_classification(estimator, disparity_moment)


def test_thresholdoptimizer_classification():
    estimator = create_model()

    ptc.run_thresholdoptimizer_classification(estimator)

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

#print("About to start bias mitigation training: \n")

#model = MLP(229)
#print("Train model.\n")
#train_model(train_dl, model)
#print("Model trained!\n")

#model.fit(train_d1.dataset.X(), train_d1.dataset.y())
#print("Commencing Gridsearch \n")

#sweep = GridSearch(model, constraints = DemographicParity, grid_size = 71)
#sweep.fit(train_d1.dataset.X(), train_d1.dataset.y(), sensitive_features = test_d1.dataset.X())

#mse = eval_model(test_dl, model)
#if(mse == 0): print("this seems wrong bud\n")
#print('MSE: %.3f, RMSE: %.3f' %(mse, sqrt(mse)))

#row = train_dl.dataset[3]
#print(row[0])
#print(len(row[0]))

#yhat = predict(row[0], model)
#print("Pred: %.3f" %yhat)

"""
