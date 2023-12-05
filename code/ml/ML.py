import pandas as pd
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold, cross_val_score, cross_validate,RepeatedKFold,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

# Load data
path = '/Users/willweng/Desktop/research work/casptone/PrimeKG/output/RGCN/PrimekG2/embeddings_result_100.csv'
df = pd.read_csv(path, sep='\t')
df = df.rename(columns={'Unnamed: 0': 'geo_accession'})

# Prepare features and labels
x = df.drop(['geo_accession', 'label'], axis=1).values
y = df.label.values

# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Define stratified K-fold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

# Define models based on previous grid search results
svc = SVC(random_state=42,C = 0.1, gamma = 'scale', kernel = 'poly', probability=True)
logistic_model = LogisticRegression(random_state=42,C=0.1,solver = 'newton-cg')
rf  = RandomForestClassifier(random_state=42, max_depth = 5, n_estimators =  100,)
xgb = XGBClassifier(random_state=42, learning_rate = 0.01, max_depth = 5, n_estimators = 50)


models = {
    'Support Vector Machine':svc,
    'Logistic Regression':logistic_model,
    'Random Forest':rf,
    'XGboost':xgb
}
scoring_metrics = ['roc_auc','accuracy','recall','f1']
results = {}
for model_name,model in models.items():
    results[model_name] = {}
    for metric in scoring_metrics:
        scores = cross_val_score(model, x_scaled, y, scoring=metric, cv=cv)
        results[model_name][metric] = list(scores)

file_path = '/Users/willweng/Desktop/research work/casptone/PrimeKG/output/RGCN/ml_validation_results.json'
with open(file_path,'w') as f:
    json.dump(results, f, indent=4)


# # Fit models and evaluate on the test set
# test_results = {}
# for model_name, model in models.items():
#     model.fit(x_train, y_train)
#     test_predictions = model.predict(x_test)
#     # For binary classification, use [:, 1], for multi-class, use predict_proba(x_test) directly
#     test_probabilities = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
#     precision, recall, thresholds = precision_recall_curve(y_test, test_probabilities)
#     auc_score = auc(recall, precision)
#     print(f"AUC of Precision-Recall Curve: {auc_score}")
#     test_results[model_name] = {
#         'accuracy': accuracy_score(y_test, test_predictions),
#         'precision': precision_score(y_test, test_predictions,zero_division=0),
#         'recall': recall_score(y_test, test_predictions),
#         'f1': f1_score(y_test, test_predictions),
#         'roc_auc': roc_auc_score(y_test, test_probabilities) if test_probabilities is not None else None,
#         'roc-pr':auc_score
#     }



# sklearn.metrics.SCORERS.keys()
# file_path = '/Users/willweng/Desktop/research work/casptone/PrimeKG/output/RGCN/ml_results.json'
# with open(file_path,'w') as f:
#     json.dump(test_results, f, indent=4)






