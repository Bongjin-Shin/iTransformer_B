# import pandas as pd

# # 1. 컬럼명 정의 (이미지 내용 기준)
# col_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]

# # 2. 데이터 불러오기 (공백 구분자 '\s+' 사용)
# df = pd.read_csv('/home/bongja/Downloads/CMAPSSData/test_FD001.txt', sep='\s+', header=None, names=col_names)

# # 3. CSV 파일로 저장
# # index=False를 설정해야 행 번호(0, 1, 2...)가 파일에 저장되지 않습니다.
# df.to_csv('/home/bongja/Documents/test_FD001_with_columns.csv', index=False)

# print("저장이 완료되었습니다!")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

dir_path = '/home/bongja/Downloads/CMAPSSData/'
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1,22)] 
col_names = index_names + setting_names + sensor_names
train = pd.read_csv((dir_path+'train_FD001.txt'), sep='\s+', header=None, names=col_names)


delete_columns = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
for s in delete_columns:
    del train[s]

# print(train.head())

for en in [82]:
    plt.figure(figsize=(16, 3))
    plt.title('FD001 train engine %d Sensor'%(en))
    engine = train[train['unit_nr']==en]
    index = engine.index.tolist()
    for s in engine.columns.tolist()[10:11]:
        val = np.array(engine[s].values, dtype=np.float32)
        val = val.reshape((-1, 1))
        mx = MinMaxScaler()    
        val = mx.fit_transform(val)
        plt.plot(index, val, alpha=.6, marker='.')
    plt.grid()
    plt.show()
