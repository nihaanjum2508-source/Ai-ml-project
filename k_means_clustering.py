import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv("easy_queue_data.csv")

X = data[['wait_time', 'service_time']]

kmeans = KMeans(n_clusters=2, random_state=42)
data['cluster'] = kmeans.fit_predict(X)

print(data.head())
print("Cluster Centers:")
print(kmeans.cluster_centers_)
