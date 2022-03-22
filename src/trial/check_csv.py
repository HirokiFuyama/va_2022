import pandas as pd
import matplotlib.pyplot as plt

path_a = rf'../../data/process/fix_a_1000.csv'
path_v = rf'../../data/process/fix_v_1000.csv'

df_a = pd.read_csv(path_a)
df_v = pd.read_csv(path_v)

df_a["time"] = [i.split("\\")[-1][:8] for i in df_a['path']]
df_v["time"] = [i.split("\\")[-1][:8] for i in df_v['path']]
df_a["time"] = pd.to_datetime(df_a["time"])
df_v["time"] = pd.to_datetime(df_v["time"])

plt.figure(figsize=(20, 5))
plt.plot(df_a['time'], df_a['rmse'], alpha=0.7)
plt.show()

plt.figure(figsize=(20, 5))
plt.plot(df_v["time"], df_v['rmse'], alpha=0.7)
plt.show()
