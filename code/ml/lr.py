import pandas as pd
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
import numpy as np

# Load data
df = pd.read_csv('/Users/willweng/Desktop/research work/casptone/PrimeKG/output/RGCN/PrimekG/embeddings_result_64.csv', sep='\t')
df = df.rename(columns={'Unnamed: 0': 'geo_accession'})

# Prepare features and labels
x = df.drop(['geo_accession', 'label'], axis=1).values
y = df.label.values

# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# # Perform PCA to maintain 90% of the information
# pca = PCA(n_components=0.9)
# x_pca = pca.fit_transform(x_scaled)

# Define stratified K-fold
repeated_stratified_kfold = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)

# Define models
svc = SVC(random_state=42,C = 1, gamma = 'scale', kernel = 'sigmoid')
logistic_model = LogisticRegression(random_state=42,C=0.1,solver = 'newton-cg')
rf  = RandomForestClassifier(random_state=42, max_depth = 3, n_estimators =  200)
xgb = XGBClassifier(random_state=42, learning_rate = 0.02, max_depth = 3, n_estimators = 100)

# Define scoring metrics
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, zero_division=0),
           'recall': make_scorer(recall_score, zero_division=0),
           'f1_score': make_scorer(f1_score, zero_division=0),
           'auc_score':make_scorer(roc_auc_score)
           }


# Perform cross-validation for SVC
svc_scores = cross_validate(svc, x_scaled, y, cv=repeated_stratified_kfold, scoring=scoring,return_train_score=True)
print("SVC Scores:", svc_scores)

# Perform cross-validation for Logistic Regression
logistic_scores = cross_validate(logistic_model, x_scaled, y, cv=repeated_stratified_kfold, scoring=scoring,return_train_score=True)
print("Logistic Regression Scores:", logistic_scores)

rf_scores = cross_validate(rf, x_scaled, y, cv=repeated_stratified_kfold, scoring=scoring,return_estimator=False,return_train_score=True)
print("Random Forest Scores:", rf_scores)

xgb_scores = cross_validate(xgb, x_scaled, y, cv=repeated_stratified_kfold, scoring=scoring,return_estimator=False,return_train_score=True)
print("Xgboost Scores:", xgb_scores)


for model_score in (svc_scores,logistic_scores,rf_scores,xgb_scores):
    mean_test_auc = np.mean(model_score['test_auc_score'])
    print(f'--------mean auc-roc score on test set is:{mean_test_auc}')


# ways to improve: (including in discussion)
# dataset larger?
# scoring ecdf 5 -->1   2
# model training embedding size 64 --> 200
# split the data may be the major one(currently random sampling) 1
# pykeen --> dgl
# batch effect -->potentially makes the dataset larger
# imblanced dataset --> smote, upsampling, downsampling
# include the part of using original gene expression value, doing z-score


