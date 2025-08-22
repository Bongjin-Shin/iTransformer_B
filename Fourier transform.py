import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일 불러오기
df = pd.read_csv('./dataset/Appliances/energy.csv')
# df = pd.read_csv('./dataset/electricity/electricity.csv')
# df = pd.read_csv('./dataset/ETT-small/ETTh1.csv')
# df = pd.read_csv('./dataset/ETT-small/ETTh2.csv')
# df = pd.read_csv('./dataset/ETT-small/ETTm1.csv')
# df = pd.read_csv('./dataset/ETT-small/ETTm2.csv')
# df = pd.read_csv('./dataset/exchange_rate/exchange_rate.csv')
# df = pd.read_csv('./dataset/traffic/traffic.csv')
# df = pd.read_csv('./dataset/weather/weather.csv')

# 'OT' 열을 선택하여 푸리에 변환 수행
# 'OT'는 'Operating Time'으로 추정되며, 시계열 분석에 적합한 데이터입니다.
signal = df['OT'].values
fft_result = np.fft.fft(signal)

# 주파수 계산
n = len(signal)
# 'date' 열을 datetime 형식으로 변환하여 데이터의 시간 간격을 계산합니다.
df['date'] = pd.to_datetime(df['date'])
time_diff = (df['date'].iloc[1] - df['date'].iloc[0]).total_seconds()

# 주파수 배열 생성
freq = np.fft.fftfreq(n, d=time_diff)

# 양수 주파수 영역만 선택하여 시각화
pos = freq > 0
positive_freq = freq[pos]
positive_power_spectrum = np.abs(fft_result[pos])

# ====== 시각화 ======
# (1) 원본 신호 전체
# plt.figure(figsize=(12, 6))
# plt.plot(signal, label='Original Signal')
# plt.xlabel('Time Step (1 step = 10 minutes)')
# plt.ylabel('Value')
# plt.title('Original Signal of OT - Full Range')
# plt.legend()
# plt.grid(True)

# (2) 원본 신호 확대 (예: 1000~2000 step)
# plt.figure(figsize=(12, 6))
# plt.plot(range(1000, 2000), signal[1000:2000], label='Original Signal (Zoomed)')
# plt.xlabel('Time Step (1 step = 10 minutes)')
# plt.ylabel('Value')
# plt.title('Original Signal of OT - Zoomed In')
# plt.legend()
# plt.grid(True)

# (3) FFT 전체
# plt.figure(figsize=(12, 6))
# plt.plot(positive_freq, positive_power_spectrum, label='FFT of OT')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.title('FFT of OT - Full Range')
# plt.legend()
# plt.grid(True)

# (4) FFT 확대 (예: 0 ~ 0.0005 Hz)
plt.figure(figsize=(12, 6))
plt.plot(positive_freq, positive_power_spectrum, label='FFT of OT (Zoomed)')
plt.xlim(-0.000001, 0.00005)  # 주파수 축 제한
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of OT (Zoomed In 0 ~ 0.00005 Hz)')
plt.legend()
plt.grid(True)

plt.show()