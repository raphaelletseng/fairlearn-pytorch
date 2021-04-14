import fairlearn.metrics as flm
import sklearn.metrics as skm
from fairlearn.metrics import true_positive_rate
from fairlearn.metrics import MetricFrame
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from more_itertools import locate
from functools import reduce

def test(args, model, device, X_test, y_test, sensitive_idx):
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    correct = 0
    i = 0

    avg_revall = 0
    avg_precision = 0
    overall_results = []
    avg_eq_odds
