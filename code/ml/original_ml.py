import pandas as pd
from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold, cross_val_score, cross_validate,RepeatedKFold,train_test_split
from sklearn.metrics import precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from xgboost import XGBClassifier
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

path = '/Users/willweng/Desktop/research work/casptone/PrimeKG/output/scoring-5.csv'

scoring = pd.read_csv(path,sep='\t',index_col=0)
scoring = scoring.loc[:,['geo_accession','label']]
metadata = pd.read_csv('/Users/willweng/Desktop/research work/casptone/PrimeKG/output/metadata_2.csv',sep='\t')

columns =  metadata['GENE_SYMBOL']
index_list = metadata.columns[1:]

df = pd.DataFrame(index = index_list,columns=columns,data = metadata.iloc[:,1:].values.T)
df.reset_index(inplace=True)
df.rename({'index':'geo_accession'},axis=1,inplace=True)
df = df.merge(scoring,on='geo_accession',how='right')


x = df.drop(['geo_accession','label'],axis=1).values
y = df['label'].values

# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Define stratified K-fold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

svc = SVC(random_state=42,C = 10, gamma = 'auto', kernel = 'rbf', probability=True)
logistic_model = LogisticRegression(random_state=42,C=0.1,solver = 'newton-cg')
rf  = RandomForestClassifier(random_state=42, max_depth = 3, n_estimators =  200,)
xgb = XGBClassifier(random_state=42, learning_rate = 0.05, max_depth = 5, n_estimators = 200)


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
        scores = cross_val_score(model, x_train, y_train, scoring=metric, cv=cv)
        results[model_name][metric] = list(scores)


file_path = '/Users/willweng/Desktop/research work/casptone/PrimeKG/output/raw/raw_ml_validation_results.json'
with open(file_path,'w') as f:
    json.dump(results, f, indent=4)

# Fit models and evaluate on the test set
test_results = {}
for model_name, model in models.items():
    model.fit(x_train, y_train)
    test_predictions = model.predict(x_test)
    # For binary classification, use [:, 1], for multi-class, use predict_proba(x_test) directly
    test_probabilities = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None
    precision, recall, thresholds = precision_recall_curve(y_test, test_probabilities)
    auc_score = auc(recall, precision)
    print(f"AUC of Precision-Recall Curve: {auc_score}")
    test_results[model_name] = {
        'accuracy': accuracy_score(y_test, test_predictions),
        'precision': precision_score(y_test, test_predictions,zero_division=0),
        'recall': recall_score(y_test, test_predictions),
        'f1': f1_score(y_test, test_predictions),
        'roc_auc': roc_auc_score(y_test, test_probabilities) if test_probabilities is not None else None,
        'roc-pr':auc_score
    }

file = '/Users/willweng/Desktop/research work/casptone/PrimeKG/output/raw/raw_ml_results.json'
with open(file) as f:
    test_results = json.load(f)

data = {}
for model in test_results.keys():
    score = test_results[model]['roc_auc']
    data[model] = [score]


df_auc = pd.DataFrame.from_dict(data)
out = '/Users/willweng/Desktop/research work/casptone/PrimeKG/output/raw/plot/'
plt.figure(figsize=(10, 6))
sns.barplot(data=df_auc)
plt.title('Raw Data', fontsize=15)
plt.ylabel('AUC-ROC Score', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.grid(axis='y')
plt.savefig(out + 'auc_roc_score.jpg')
plt.show()








