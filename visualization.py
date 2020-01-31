import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# loading dataset into Pandas DataFrame
# df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

df = pd.read_csv("combined_csv.csv", names=["Open", "High", "Low", "Close", "Volume", "Ex-Dividend", "Split", "Adj Open", "Adj High", "Adj Low", "Adj Close", "Adj Volume", "Company"])

from sklearn.preprocessing import StandardScaler

features = ["Open", "High", "Low", "Close", "Volume", "Ex-Dividend", "Split", "Adj Open", "Adj High", "Adj Low", "Adj Close", "Adj Volume"]
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,["Company"]].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['Company']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']
colors = ['r', 'g', 'b', 'y', 'green', 'black', "grey", "orange", "purple", "pink", "brown", "beige", "maroon", "bronze", "gold"]
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['Company'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c=color
#                , s = 50)

for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Company'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2'])
ax.legend(targets)
ax.grid()
plt.show()

# pca.explained_variance_ratio_
