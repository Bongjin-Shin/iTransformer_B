import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_CMAPSS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='train_FD001_with_columns.csv',
                 target='s21', scale=True, timeenc=0, freq='h',
                 engine_id=None, train_limit=100):
        
        # size: [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24
            self.label_len = 12
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_limit = train_limit

        self.root_path = root_path
        self.data_path = data_path
        
        # 1. 엔진 분할 (Train: 1~70, Val: 71~80, Test: 81~100)
        if self.flag == 'train':
            self.target_engine_ids = list(range(1, 71)) 
        elif self.flag == 'val':
            self.target_engine_ids = list(range(71, 81)) 
        elif self.flag == 'test':
            self.target_engine_ids = list(range(81, 101))
            
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 제외할 센서: s1, s5, s6, s10, s16, s18, s19
        exclude_sensors = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
        sensor_cols = [f's{i}' for i in range(1, 22)]
        sensor_cols = [col for col in sensor_cols if col in df_raw.columns and col not in exclude_sensors]
        print(f"[{self.flag.upper()}] Using sensors: {sensor_cols} (Total: {len(sensor_cols)})")
        
        self.engine_data_list = [] 
        self.valid_indices = []

        # -------------------------------------------------------
        # 2. Scaler 학습 (항상 Train 엔진 1~70 데이터 기준)
        # -------------------------------------------------------
        df_train_fit = df_raw[df_raw['id'].isin(range(1, 71))].copy()
        fit_data_list = []
        
        # 각 Train 엔진의 학습 사용 구간(100행)만 모아서 스케일링
        limit_idx = self.train_limit
        for eid in range(1, 71):
            df_e = df_train_fit[df_train_fit['id'] == eid]
            # 여기서는 reset_index가 없어도 iloc[:limit]가 절대 위치가 아닌 
            # 필터링된 데이터프레임 내 상대 위치(0~limit)를 자르므로 정확합니다.
            df_e_subset = df_e.iloc[:limit_idx]
            
            if self.features == 'M' or self.features == 'MS':
                fit_data_list.append(df_e_subset[sensor_cols].values)
            else:
                fit_data_list.append(df_e_subset[[self.target]].values)
                
        if fit_data_list:
            self.scaler.fit(np.concatenate(fit_data_list, axis=0))
        
        # -------------------------------------------------------
        # 3. 실제 데이터 로드 (각 엔진별로 수행)
        # Train/Val: 100행까지만 사용 (정상 패턴 학습)
        # Test: 각 엔진의 전체 데이터 사용 (비정상 구간까지 평가)
        # -------------------------------------------------------
        print(f"[{self.flag.upper()}] Loading Engines: {self.target_engine_ids[0]} ~ {self.target_engine_ids[-1]}")
        
        for i, eid in enumerate(self.target_engine_ids):
            # [핵심] 엔진 ID로 필터링 -> 해당 엔진 데이터만 남음
            df_e = df_raw[df_raw['id'] == eid].copy()
            
            # [핵심] 인덱스 초기화 -> 0번 행이 해당 엔진의 시작점이 됨
            df_e = df_e.reset_index(drop=True)
            
            # Train/Val: 100행까지만, Test: 전체 데이터 사용
            if self.flag == 'test':
                end_row = len(df_e)  # 테스트는 엔진 전체 데이터 사용
            else:
                end_row = self.train_limit  # 학습/검증은 100행까지만
            
            if self.features == 'M' or self.features == 'MS':
                df_data = df_e[sensor_cols]
            else:
                df_data = df_e[[self.target]]
            
            raw_data = df_data.values[:end_row]
            
            if self.scale:
                data = self.scaler.transform(raw_data)
            else:
                data = raw_data
                
            cycles = df_e['cycle'].values[:end_row]
            norm_base = df_e['cycle'].max() 
            data_stamp = np.zeros((len(cycles), 4))
            data_stamp[:, 0] = cycles / norm_base
            
            self.engine_data_list.append((data, data, data_stamp))
            
            # 유효 인덱스 계산
            max_idx = len(data) - self.seq_len - self.pred_len
            min_idx = max(0, self.label_len - self.seq_len)
            
            if max_idx >= min_idx:
                for start_idx in range(min_idx, max_idx + 1):
                     self.valid_indices.append((i, start_idx))

        print(f"[{self.flag.upper()}] Total Valid Samples: {len(self.valid_indices)}")

    def __getitem__(self, index):
        eng_idx, s_begin = self.valid_indices[index]
        data_x, data_y, data_stamp = self.engine_data_list[eng_idx]
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = data_x[s_begin:s_end]
        seq_y = data_y[r_begin:r_end]
        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.valid_indices)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_CMAPSS_Expanding(Dataset):
    """
    Dynamic/Expanding Lookback Window Dataset for C-MAPSS
    
    Unlike fixed sliding window, this dataset expands the lookback window as sliding progresses:
    - Sample 0: lookback = [0, init_seq_len), predict = [init_seq_len, init_seq_len + pred_len)
    - Sample 1: lookback = [0, init_seq_len + 1), predict = [init_seq_len + 1, init_seq_len + 1 + pred_len)
    - Sample k: lookback = [0, init_seq_len + k), predict = [init_seq_len + k, init_seq_len + k + pred_len)
    
    The lookback window always starts from 0 and expands as we slide forward.
    Padding is applied to match max_seq_len for batch processing.
    """
    
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='train_FD001_with_columns.csv',
                 target='s21', scale=True, timeenc=0, freq='h',
                 engine_id=None, train_limit=100, 
                 init_seq_len=48, max_seq_len=200, sliding_step=1):
        
        # size: [seq_len, label_len, pred_len]
        # init_seq_len and max_seq_len are explicitly passed, not derived from size
        if size is None:
            self.label_len = 0  # Not used in expanding mode
            self.pred_len = 24
        else:
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Use explicit init_seq_len and max_seq_len parameters
        self.init_seq_len = init_seq_len
        self.max_seq_len = max_seq_len
        self.sliding_step = sliding_step
        
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_limit = train_limit

        self.root_path = root_path
        self.data_path = data_path
        
        # Engine split (Train: 1~70, Val: 71~80, Test: 81~100)
        if self.flag == 'train':
            self.target_engine_ids = list(range(1, 71)) 
        elif self.flag == 'val':
            self.target_engine_ids = list(range(71, 81)) 
        elif self.flag == 'test':
            self.target_engine_ids = list(range(81, 101))
            
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # Excluded sensors: s1, s5, s6, s10, s16, s18, s19
        exclude_sensors = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
        sensor_cols = [f's{i}' for i in range(1, 22)]
        sensor_cols = [col for col in sensor_cols if col in df_raw.columns and col not in exclude_sensors]
        self.sensor_cols = sensor_cols
        print(f"[{self.flag.upper()}] Using sensors: {sensor_cols} (Total: {len(sensor_cols)})")
        
        self.engine_data_list = [] 
        self.valid_indices = []

        # Scaler fitting (always on Train engines 1~70)
        df_train_fit = df_raw[df_raw['id'].isin(range(1, 71))].copy()
        fit_data_list = []
        
        limit_idx = self.train_limit
        for eid in range(1, 71):
            df_e = df_train_fit[df_train_fit['id'] == eid]
            df_e_subset = df_e.iloc[:limit_idx]
            
            if self.features == 'M' or self.features == 'MS':
                fit_data_list.append(df_e_subset[sensor_cols].values)
            else:
                fit_data_list.append(df_e_subset[[self.target]].values)
                
        if fit_data_list:
            self.scaler.fit(np.concatenate(fit_data_list, axis=0))
        
        # Load data for each engine
        print(f"[{self.flag.upper()}] Loading Engines: {self.target_engine_ids[0]} ~ {self.target_engine_ids[-1]}")
        print(f"[{self.flag.upper()}] Expanding mode: init_seq_len={self.init_seq_len}, max_seq_len={self.max_seq_len}, step={self.sliding_step}")
        
        for i, eid in enumerate(self.target_engine_ids):
            df_e = df_raw[df_raw['id'] == eid].copy()
            df_e = df_e.reset_index(drop=True)
            
            # Train/Val: 100 rows only, Test: full data
            if self.flag == 'test':
                end_row = len(df_e)
            else:
                end_row = self.train_limit
            
            if self.features == 'M' or self.features == 'MS':
                df_data = df_e[sensor_cols]
            else:
                df_data = df_e[[self.target]]
            
            raw_data = df_data.values[:end_row]
            
            if self.scale:
                data = self.scaler.transform(raw_data)
            else:
                data = raw_data
                
            cycles = df_e['cycle'].values[:end_row]
            norm_base = df_e['cycle'].max() 
            data_stamp = np.zeros((len(cycles), 4))
            data_stamp[:, 0] = cycles / norm_base
            
            self.engine_data_list.append((data, data, data_stamp))
            
            # Valid indices for expanding window
            # Sample k: lookback = [0, init_seq_len + k * step), predict = [init_seq_len + k * step, ...)
            # Maximum k such that init_seq_len + k * step + pred_len <= len(data)
            data_len = len(data)
            k = 0
            while True:
                current_seq_len = self.init_seq_len + k * self.sliding_step
                pred_start = current_seq_len
                pred_end = pred_start + self.pred_len
                
                if pred_end > data_len:
                    break
                # For Test: no max_seq_len limit (predict until engine end)
                # For Train/Val: apply max_seq_len limit
                if self.flag != 'test' and current_seq_len > self.max_seq_len:
                    break
                    
                # Store (engine_idx, sample_k, actual_seq_len)
                self.valid_indices.append((i, k, current_seq_len))
                k += 1

        print(f"[{self.flag.upper()}] Total Valid Samples: {len(self.valid_indices)}")

    def __getitem__(self, index):
        eng_idx, sample_k, actual_seq_len = self.valid_indices[index]
        data_x, data_y, data_stamp = self.engine_data_list[eng_idx]
        
        # Prediction range is always based on actual_seq_len
        pred_start = actual_seq_len
        pred_end = pred_start + self.pred_len
        
        # For Test: if actual_seq_len > max_seq_len, use only the most recent max_seq_len data
        # This is "Truncation" - use recent data instead of full expanding window
        if self.flag == 'test' and actual_seq_len > self.max_seq_len:
            # Truncate: use only the most recent max_seq_len rows
            lookback_start = actual_seq_len - self.max_seq_len
            lookback_end = actual_seq_len
            use_seq_len = self.max_seq_len
        else:
            # Normal expanding: start from 0
            lookback_start = 0
            lookback_end = actual_seq_len
            use_seq_len = actual_seq_len
        
        # Get data
        seq_x_actual = data_x[lookback_start:lookback_end]  # [use_seq_len, features]
        seq_y = data_y[pred_start:pred_end]  # [pred_len, features]
        seq_x_mark_actual = data_stamp[lookback_start:lookback_end]  # [use_seq_len, 4]
        seq_y_mark = data_stamp[pred_start:pred_end]  # [pred_len, 4]
        
        # Padding to max_seq_len (always fixed size for model compatibility)
        num_features = seq_x_actual.shape[1]
        seq_x = np.zeros((self.max_seq_len, num_features))
        seq_x_mark = np.zeros((self.max_seq_len, 4))
        
        # Right-align the actual data (padding at the beginning)
        pad_len = self.max_seq_len - use_seq_len
        seq_x[pad_len:] = seq_x_actual
        seq_x_mark[pad_len:] = seq_x_mark_actual
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.valid_indices)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def get_actual_seq_len(self, index):
        """Return the actual sequence length for a given sample index"""
        _, _, actual_seq_len = self.valid_indices[index]
        return actual_seq_len


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
