import matplotlib.pyplot as plt

df['execution_time'] = pd.to_datetime(df['execution_time'])
df.set_index('execution_time', inplace=True)
df['price'].plot(title="Trade Prices Over Time")
plt.show()
