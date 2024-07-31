import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

warnings.filterwarnings('ignore')

start = time.time()
df = pd.read_csv('dataset\Spatial-Clustering_ctr_mag4_5upper.csv')
final_centr = df.groupby('cluster')[['longitude', 'latitude']].mean().reset_index()

plt.figure(figsize=(15,8))
sns.scatterplot(x=df['longitude'], y=df['latitude'], hue=df['final_cluster'], palette='viridis')
sns.scatterplot(x='longitude', y='latitude', data=final_centr, color='red', marker='X', s=20, label='Centroid')
# print(time.time() - start)
plt.show()