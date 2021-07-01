import pandas as pd

iris = pd.read_csv('IRIS.csv')
# print(iris.head(5))

# split data attributes and label attribute
attributes = iris.drop(['species'], axis=1)
labels = iris['species']

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(attributes)

# find/predict the clusters list for the dataset
y_pred = model.predict(attributes)
print(y_pred)


# evaluation stage
from sklearn import metrics
contingecyMatric = metrics.cluster.contingency_matrix(labels, y_pred)
print(contingecyMatric)

ari = metrics.cluster.adjusted_rand_score(labels, y_pred)
print('ari', ari)
nmi = metrics.cluster.normalized_mutual_info_score(labels, y_pred)
print('nmi', nmi)