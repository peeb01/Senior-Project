import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('juain_date405.csv').tail(500)
df['time'] = pd.to_datetime(df['time'])
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['hour'] = df['time'].dt.strftime('%H')
df['minute'] = df['time'].dt.strftime('%M')

# def windows(windowsize=5):
#     x
# plt.plot(df['day'])
# plt.plot(df['hour'])
plt.plot(df['minute'])
# plt.plot(df['jd'] - 2.46e6)

plt.show()