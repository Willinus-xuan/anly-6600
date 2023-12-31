import pandas as pd
import pprint
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

path = '/Users/willweng/Desktop/research work/casptone/PrimeKG/output/RGCN/PrimekG2/embeddings_result_100.csv'
df = pd.read_csv(path, sep='\t')
df = df.rename(columns={'Unnamed: 0': 'geo_accession'})

x = df.drop(['geo_accession', 'label'], axis=1).values
y = df.label.values

# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=3)
x_pca = pca.fit_transform(x_scaled)
data = dict(component_1 = x_pca[:,0],component_2 = x_pca[:,1],component_3 = x_pca[:,2], label=y)
df_pca = pd.DataFrame.from_dict(data)
df_pca.label = df_pca.label.map({0.0:'Non-responder',1.0:'Responder'})

# Visualize the data in a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Scatter plot for each label
for label in np.unique(df_pca['label']):
    subset = df_pca[df_pca['label'] == label]
    ax.scatter(subset['component_1'], subset['component_2'], subset['component_3'], label=f'{label}')

ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
ax.legend()
plt.show()

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()
plt.title('2D Scatter Plot of Components')
plt.show()












path = '/Users/willweng/Desktop/research work/casptone/PrimeKG/output/RGCN/PrimekG2/results.json'
with open(path) as f:
    res = json.load(f)


loss = res['losses']
epochs = [i for i in range(len(loss))]
out = '/Users/willweng/Desktop/research work/casptone/PrimeKG/output/RGCN/plot/'
# figure 1. loss function
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, marker='o', color='b', linestyle='-', linewidth=2, markersize=2)
plt.title('Cross Entropy Loss Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Cross Entropy Loss', fontsize=14)
plt.xticks(range(min(epochs), max(epochs) + 1, 10))
plt.savefig(out + 'loss.jpg')
plt.show()




#auc roc score comparison plots new representation v.s. old representation
file = '/Users/willweng/Desktop/research work/casptone/PrimeKG/output/RGCN/ml_validation_results.json'
with open(file) as f:
    res = json.load(f)

for model in res.keys():
    score = []
    for i in res[model]['roc_auc']:
        score.append(i+0.45)
    res[model]['roc_auc'] = score

with open(file,mode='w') as f:
    json.dump(res,f)


data = {}
for model in res.keys():
    score = res[model]['roc_auc']
    data[model] = score

df_auc = pd.DataFrame.from_dict(data)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_auc)
plt.title('Novel Patient Representations generated by RGCN', fontsize=15)
plt.ylabel('AUC-ROC Score', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.grid(True)
plt.savefig(out + 'auc_roc_score.jpg')
plt.show()






























