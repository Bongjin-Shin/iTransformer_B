import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# CSV 파일 경로
file_path = "/home/bongja/GitHub_Code/iTransformer/dataset/Battery/B0005_discharge_detailed.csv"

# 1. CSV 불러오기
df = pd.read_csv(file_path)
cols = list(df.columns)
cols.remove('OT')
cols.remove('date')
df = df[['date'] + cols + ['OT']]  # 'date' 맨 앞으로, 'OT' 맨 뒤로
# 2. 마지막 채널 선택 (첫 열이 time이라고 가정하고 제외)
num_train = int(len(df) * 0.7)
num_test = int(len(df) * 0.2)

cols_data = df.columns[1:]
df_data = df[cols_data]

train_data = df_data[0 : num_train]

# 3. 전체 데이터 정규화
scaler = StandardScaler()
scaler.fit(train_data.values)
scaled_values = scaler.transform(df_data.values)

# 4. 정규화된 마지막 채널
scaled_last_channel = scaled_values[:, -1]
target_data = scaled_last_channel[len(df) - num_test : len(df)]

# 5. 그래프 그리기
plt.figure(figsize=(12, 4))
plt.plot(target_data, label=df.columns[-1])
plt.title("Normalized Last Channel")
plt.xlabel("Time Index")
plt.ylabel("Scaled Value")
plt.legend()
plt.show()