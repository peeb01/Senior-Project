import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

warnings.filterwarnings('ignore')

# # start = time.time()
# df = pd.read_csv('dataset\Spatial-Clustering_ctr_mag4_5upper.csv')
# df = df[df['mag']>=4.5]

# n_data = pd.read_csv('dataset\\all.csv')
# final_centr = df.groupby('cluster')[['longitude', 'latitude']].mean().reset_index()
# # print(df['final_cluster'].value_counts())
# plt.figure(figsize=(15,8))
# sns.scatterplot(x=df['longitude'], y=df['latitude'])
# # sns.scatterplot(x='longitude', y='latitude', data=final_centr, color='red', marker='X', s=20, label='Centroid')
# # print(time.time() - start)

# plt.scatter(n_data['lon'], n_data['lat'])
# plt.show()

