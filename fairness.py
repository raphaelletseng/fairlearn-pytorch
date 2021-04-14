from sklearn.model_selection import train_test_split
from fairlearn.reductions import GridSearch
from fairlearn.reductions import DemographicParity, ErrorRate

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

from sklearn.datasets import fetch_openml
import sklearn.metrics as skm

import wandb
wandb.login()
run_name = "fairlearn-sex-nodp-run1"

data = fetch_openml(data_id = 1590, as_frame = True)
X_raw = data.data
Y = (data.target == '>50K') * 1

A = X_raw["sex"]
X = X_raw.drop(labels = ['sex'], axis = 1)
#print(X)

X = pd.get_dummies(X) #one hot

print("X has been encoded.\n")
#print(len(X))
#print(X)

sc = StandardScaler()
X_scaled = sc.fit_transform(X);
X_scaled = pd.DataFrame(X_scaled, columns = X.columns)

le = LabelEncoder()
Y = le.fit_transform(Y)

#print(f"TYpe of X: {type(X_scaled)} \n")
#print(type(Y))
#print(f"Type of A: {type(A)} \n")

wandb.init(project="fairlearn-pytorch-example-notebook", name = run_name, config={
    "run_name": run_name,
    "architecture": "Sklearn LogisticRegression",
    "dataset": "adult",
})
config = wandb.config

print("Y has been tranformed. Assigning sets.\n")

X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(
    X_scaled, Y, A, test_size = 0.2, random_state = 0, stratify = Y
)
#

X_train = X_train.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
X_test = X_test.reset_index(drop = True)
A_test = A_test.reset_index(drop=True)

print("About to start bias mitigation training.\n")

unmitigated_predictor = LogisticRegression(solver = 'liblinear', fit_intercept = True)
unmitigated_predictor.fit(X_train, Y_train)

#--------- GRID SEARCH -----------------------#
print("Commencing GridSearch.\n")

sweep = GridSearch(LogisticRegression(solver='liblinear', fit_intercept=True),
                        constraints = DemographicParity(),
                        grid_size=71)

sweep.fit(X_train, Y_train, sensitive_features = A_train)
predictors = sweep.predictors_

errors, disparities, accuracies = [], [], []
for m in predictors:
    def classifier(X): return m.predict(X)
    error = ErrorRate()
    print(error, "\n")
    error.load_data(X_train, pd.Series(Y_train), sensitive_features=A_train)
    disparity = DemographicParity()
    disparity.load_data(X_train, pd.Series(Y_train), sensitive_features= A_train)

    errors.append(error.gamma(classifier)[0])
    disparities.append(disparity.gamma(classifier).max())

    accuracy = skm.accuracy_score(Y_test, classifier(X_test))
    accuracies.append(accuracy)

    error_log = error.gamma(classifier)[0]
    disparity_log = disparity.gamma(classifier).max()

    wandb.log({
        'error': error_log,
        'disparity': disparity_log,
        'acc': accuracy
    })

all_results = pd.DataFrame({"predictor": predictors, "error": errors, "disparity": disparities, "accuracy": accuracies})
all_results2 = pd.DataFrame({"error": errors, "disparity": disparities, "accuracy": accuracies})

non_dominated= []
for row in all_results.itertuples():
    error_for_lower_or_eq_disparity = all_results["error"][all_results["disparity"] <= row.disparity]
    if row.error <= error_for_lower_or_eq_disparity.min():
        non_dominated.append(row.predictor)

print(all_results)
print("#################################################")
#print(non_dominated)

file_path = "fairlearn-notebook//all_results."+ run_name
file_object = open(file_path, "a+")
all_results2.to_csv(file_object, index = False)
file_object.close()



#--- DASHBOARD STUFF ---
