import pandas as pd
import plotly.express as px
##############################################################################
# 데이터 불러오기
file_path = './dataset/CMAPSSData/train_FD001_with_columns.csv'
df = pd.read_csv(file_path)

# 마지막 열 (타겟 변수) 가져오기
target_column = df.iloc[:, :]

# Plotly로 인터랙티브 라인 그래프 생성
fig = px.line(data_frame=df, y=target_column, title='Appliances Data')

# 그래프 레이아웃 설정 (선택 사항)
fig.update_layout(
    xaxis_title='Time Step',
    yaxis_title='Value',
    height=700 # 그래프의 높이 조절
)

fig.show()
##############################################################################
# # 데이터 불러오기
# file_path = './dataset/electricity/electricity.csv'
# df = pd.read_csv(file_path)

# # 마지막 열 (타겟 변수) 가져오기
# target_column = df.iloc[:, -1]

# # Plotly로 인터랙티브 라인 그래프 생성
# fig = px.line(data_frame=df, y=target_column, title='Electricity Data')

# # 그래프 레이아웃 설정 (선택 사항)
# fig.update_layout(
#     xaxis_title='Time Step',
#     yaxis_title='Value',
#     height=700 # 그래프의 높이 조절
# )

# fig.show()
# ##############################################################################
# # 데이터 불러오기
# file_path = './dataset/ETT-small/ETTh1.csv'
# df = pd.read_csv(file_path)

# # 마지막 열 (타겟 변수) 가져오기
# target_column = df.iloc[:, -1]

# # Plotly로 인터랙티브 라인 그래프 생성
# fig = px.line(data_frame=df, y=target_column, title='ETTh1 Data')

# # 그래프 레이아웃 설정 (선택 사항)
# fig.update_layout(
#     xaxis_title='Time Step',
#     yaxis_title='Value',
#     height=700 # 그래프의 높이 조절
# )

# fig.show()
# ##############################################################################
# # 데이터 불러오기
# file_path = './dataset/ETT-small/ETTh2.csv'
# df = pd.read_csv(file_path)

# # 마지막 열 (타겟 변수) 가져오기
# target_column = df.iloc[:, -1]

# # Plotly로 인터랙티브 라인 그래프 생성
# fig = px.line(data_frame=df, y=target_column, title='ETTh2 Data')

# # 그래프 레이아웃 설정 (선택 사항)
# fig.update_layout(
#     xaxis_title='Time Step',
#     yaxis_title='Value',
#     height=700 # 그래프의 높이 조절
# )

# fig.show()
# ##############################################################################
# # 데이터 불러오기
# file_path = './dataset/ETT-small/ETTm1.csv'
# df = pd.read_csv(file_path)

# # 마지막 열 (타겟 변수) 가져오기
# target_column = df.iloc[:, -1]

# # Plotly로 인터랙티브 라인 그래프 생성
# fig = px.line(data_frame=df, y=target_column, title='ETTm1 Data')

# # 그래프 레이아웃 설정 (선택 사항)
# fig.update_layout(
#     xaxis_title='Time Step',
#     yaxis_title='Value',
#     height=700 # 그래프의 높이 조절
# )

# fig.show()
# ##############################################################################
# # 데이터 불러오기
# file_path = './dataset/ETT-small/ETTm2.csv'
# df = pd.read_csv(file_path)

# # 마지막 열 (타겟 변수) 가져오기
# target_column = df.iloc[:, -1]

# # Plotly로 인터랙티브 라인 그래프 생성
# fig = px.line(data_frame=df, y=target_column, title='ETTm2 Data')

# # 그래프 레이아웃 설정 (선택 사항)
# fig.update_layout(
#     xaxis_title='Time Step',
#     yaxis_title='Value',
#     height=700 # 그래프의 높이 조절
# )

# fig.show()
# ##############################################################################
# # 데이터 불러오기
# file_path = './dataset/exchange_rate/exchange_rate.csv'
# df = pd.read_csv(file_path)

# # 마지막 열 (타겟 변수) 가져오기
# target_column = df.iloc[:, -1]

# # Plotly로 인터랙티브 라인 그래프 생성
# fig = px.line(data_frame=df, y=target_column, title='Exchange_rate Data')

# # 그래프 레이아웃 설정 (선택 사항)
# fig.update_layout(
#     xaxis_title='Time Step',
#     yaxis_title='Value',
#     height=700 # 그래프의 높이 조절
# )

# fig.show()
# ##############################################################################
# # 데이터 불러오기
# file_path = './dataset/traffic/traffic.csv'
# df = pd.read_csv(file_path)

# # 마지막 열 (타겟 변수) 가져오기
# target_column = df.iloc[:, -1]

# # Plotly로 인터랙티브 라인 그래프 생성
# fig = px.line(data_frame=df, y=target_column, title='Traffic Data')

# # 그래프 레이아웃 설정 (선택 사항)
# fig.update_layout(
#     xaxis_title='Time Step',
#     yaxis_title='Value',
#     height=700 # 그래프의 높이 조절
# )

# fig.show()
# ##############################################################################
# # 데이터 불러오기
# file_path = './dataset/weather/weather.csv'
# df = pd.read_csv(file_path)

# # 마지막 열 (타겟 변수) 가져오기
# target_column = df.iloc[:, -1]

# # Plotly로 인터랙티브 라인 그래프 생성
# fig = px.line(data_frame=df, y=target_column, title='Weather Data')

# # 그래프 레이아웃 설정 (선택 사항)
# fig.update_layout(
#     xaxis_title='Time Step',
#     yaxis_title='Value',
#     height=700 # 그래프의 높이 조절
# )

# fig.show()