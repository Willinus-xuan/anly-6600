import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold,RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from xgboost import XGBClassifier
import numpy as np

# Load data
path = '/Users/willweng/Desktop/research work/casptone/PrimeKG/output/RGCN/PrimekG/embeddings_result_100.csv'
df = pd.read_csv(path, sep='\t')
df = df.rename(columns={'Unnamed: 0': 'geo_accession'})


# Prepare features and labels
x = df.drop(['geo_accession', 'label'], axis=1).values
y = df.label.values

# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# Define stratified K-fold
stratified_kfold = RepeatedStratifiedKFold(n_splits=5,  n_repeats=5,  random_state=42)

# Define parameter grid for SVC
svc_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel':['poly','linear','rbf','sigmoid']
    # Include other parameters here
}

# Define parameter grid for Logistic Regression
logistic_param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
}

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10,20,50],
    # Include other parameters here
}

# Define parameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10,15],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    # Include other parameters here
}

# Define scoring metrics
# scoring_metrics = {
#     'accuracy': make_scorer(accuracy_score),
#     'precision': make_scorer(precision_score, average='weighted', zero_division=0),
#     'recall': make_scorer(recall_score, average='weighted', zero_division=0),
#     'f1_score': make_scorer(f1_score, average='weighted', zero_division=0),
#     'auc': 'roc_auc',  # ROC AUC does not need a make_scorer wrapper
# }

scoring_metrics = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'f1_score': make_scorer(f1_score, zero_division=0),
    'auc': 'roc_auc',  # ROC AUC does not need a make_scorer wrapper
}

# Initialize GridSearchCV objects for each classifier
classifiers = {
    'SVC': (SVC(random_state=42, kernel='sigmoid'), svc_param_grid),
    'LogisticRegression': (LogisticRegression(random_state=42), logistic_param_grid),
    'RandomForest': (RandomForestClassifier(random_state=42), rf_param_grid),
    'XGBoost': (XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), xgb_param_grid),
}

# Perform Grid Search with Cross-Validation for each classifier
for clf_name, (clf, param_grid) in classifiers.items():
    grid_search = GridSearchCV(clf, param_grid, cv=stratified_kfold, scoring=scoring_metrics, refit='auc', return_train_score=True)
    grid_search.fit(x_scaled, y)
    print(f"{clf_name} Best parameters:", grid_search.best_params_)
    print(f"{clf_name} Best score:", grid_search.best_index_)
    idx = np.argmax(grid_search.cv_results_['mean_test_auc'])
    print(f"{clf_name} Best test accuracy score:", grid_search.cv_results_['mean_test_accuracy'][idx])
    print(f"{clf_name} Best test precision score:", grid_search.cv_results_['mean_test_precision'][idx])
    print(f"{clf_name} Best test recall score:", grid_search.cv_results_['mean_test_recall'][idx])
    print(f"{clf_name} Best test f-1 score:", grid_search.cv_results_['mean_test_f1_score'][idx])
    print(f"{clf_name} Best test auc-roc score:", grid_search.cv_results_['mean_test_auc'][idx])





