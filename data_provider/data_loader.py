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
            self.target_engine_ids = list(range(1, 61))  # 기존 1~70이었는데 60으로 수정
        elif self.flag == 'val':
            self.target_engine_ids = list(range(61, 81))  # 기존 71~80이었는데 61~80으로 수정
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
        df_train_fit = df_raw[df_raw['id'].isin(range(1, 61))].copy()  # 기존 1~70이었는데 60으로 수정
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


class Dataset_CMAPSS_NormalOnly(Dataset):
    """
    C-MAPSS Dataset for Normal Region Only (1~100 엔진 모두 100행까지만)
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='train_FD001_with_columns.csv',
                 target='s21', scale=True, timeenc=0, freq='h',
                 engine_id=None, train_limit=100):
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
        # 엔진 분할 (Train: 1~70, Val: 71~80, Test: 81~100)
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
        exclude_sensors = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
        sensor_cols = [f's{i}' for i in range(1, 22)]
        sensor_cols = [col for col in sensor_cols if col in df_raw.columns and col not in exclude_sensors]
        print(f"[{self.flag.upper()}-NORMAL] Using sensors: {sensor_cols} (Total: {len(sensor_cols)})")
        self.engine_data_list = []
        self.valid_indices = []
        # Always fit scaler on Train 엔진 1~70, 100행까지만
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
        print(f"[{self.flag.upper()}-NORMAL] Loading Engines: {self.target_engine_ids[0]} ~ {self.target_engine_ids[-1]}")
        for i, eid in enumerate(self.target_engine_ids):
            df_e = df_raw[df_raw['id'] == eid].copy()
            df_e = df_e.reset_index(drop=True)
            end_row = min(self.train_limit, len(df_e))  # 모든 엔진 100행까지만
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
            max_idx = len(data) - self.seq_len - self.pred_len
            min_idx = max(0, self.label_len - self.seq_len)
            if max_idx >= min_idx:
                for start_idx in range(min_idx, max_idx + 1):
                    self.valid_indices.append((i, start_idx))
        print(f"[{self.flag.upper()}-NORMAL] Total Valid Samples: {len(self.valid_indices)}")

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


class Dataset_NPY(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='A-1.npy', # 구체적인 파일명을 받음
                 target='OT', scale=True, timeenc=0, freq='h'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
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

        self.root_path = root_path
        self.file_name = data_path  # 예: 'A-1.npy'
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        # 1. 디렉토리 선택 (train 폴더 또는 test 폴더)
        if self.flag in ['train', 'val']:
            dir_name = 'train'
        else:
            dir_name = 'test'
            
        # 2. 단일 파일 로드
        file_path = os.path.join(self.root_path, dir_name, self.file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        data = np.load(file_path)
        
        # 3. 첫 번째 Feature 선택
        if len(data.shape) == 2:
            data = data[:, 0:1] # (Time, 1)
        elif len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # 4. 데이터 분할 (Train:Val = 9:1, Test는 전체)
        if self.flag == 'test':
            border1 = 0
            border2 = len(data)
        else:
            num_train = int(len(data) * 0.9)
            if self.flag == 'train':
                border1 = 0
                border2 = num_train
            elif self.flag == 'val':
                border1 = num_train
                border2 = len(data)

        # 5. 스케일링 (Train 데이터 기준으로 fit)
        if self.scale:
            if self.flag in ['train', 'val']:
                train_data_slice = data[0:int(len(data) * 0.9)]
                self.scaler.fit(train_data_slice)
            else:
                # Test만 따로 들어올 경우, Test 데이터 자체 분포로 스케일링하거나
                # (엄밀하게는 학습된 Scaler를 저장해서 불러와야 하지만, 간편한 실험을 위해 자체 fit)
                self.scaler.fit(data)
            
            data_scaled = self.scaler.transform(data)
        else:
            data_scaled = data

        self.data_x = data_scaled[border1:border2]
        self.data_y = data_scaled[border1:border2]
        self.data_stamp = np.zeros((len(self.data_x), 4)) # Time Stamp 더미

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


class Dataset_GECCO(Dataset):
    """
    GECCO Water Quality Sensor Dataset
    Fixed borders: Train(0~12000), Val(12001~15000), Test(15001~end)
    Columns: Tp, Cl, pH, Redox, Leit, Trueb, Cl_2, Fm, Fm_2, Label
    Label column is excluded from features (used for evaluation only).
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='173_GECCO_id_1_Sensor_tr_16165_1st_16265.csv',
                 target='Tp', scale=True, timeenc=0, freq='h'):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

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
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Sensor columns (exclude Label)
        sensor_cols = [c for c in df_raw.columns if c != 'Label']
        print(f"[GECCO-{self.flag.upper()}] Sensor columns: {sensor_cols} (Total: {len(sensor_cols)})")
        print(f"[GECCO-{self.flag.upper()}] Total data length: {len(df_raw)}")

        # Fixed borders
        #   Train: 0 ~ 12000 (12001 rows)
        #   Val:   12001 ~ 15000 (3000 rows), with seq_len lookback
        #   Test:  15001 ~ end, with seq_len lookback
        num_train = 12001
        num_val_end = 15001
        border1s = [0, num_train - self.seq_len, num_val_end - self.seq_len]
        border2s = [num_train, num_val_end, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        print(f"[GECCO-{self.flag.upper()}] Data range: [{border1}, {border2}) ({border2 - border1} rows)")

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[sensor_cols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Scale using train data only
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # Dummy time stamp (no date column in GECCO)
        self.data_stamp = np.zeros((len(self.data_x), 4))

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


class Dataset_PSM(Dataset):
    """
    PSM Pooled Server Metrics Dataset
    Fixed borders: Train(0~50000), Val(50001~60000), Test(60001~end)
    Columns: feature_0 to feature_24, Label
    Label column is excluded from features (used for evaluation only).
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='115_PSM_id_1_Facility_tr_50000_1st_129872.csv',
                 target='feature_0', scale=True, timeenc=0, freq='h'):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

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
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Sensor columns (exclude Label)
        sensor_cols = [c for c in df_raw.columns if c != 'Label']
        print(f"[PSM-{self.flag.upper()}] Sensor columns: {sensor_cols[0]} ... {sensor_cols[-1]} (Total: {len(sensor_cols)})")
        print(f"[PSM-{self.flag.upper()}] Total data length: {len(df_raw)}")

        # Fixed borders
        #   Train: 0 ~ 50000 (50001 rows)
        #   Val:   50001 ~ 60000 (10000 rows)
        #   Test:  60001 ~ end
        num_train = 50001
        num_val_end = 60001
        border1s = [0, num_train - self.seq_len, num_val_end - self.seq_len]
        border2s = [num_train, num_val_end, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        print(f"[PSM-{self.flag.upper()}] Data range: [{border1}, {border2}) ({border2 - border1} rows)")

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[sensor_cols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Scale using train data only
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # Dummy time stamp (no date column in PSM)
        self.data_stamp = np.zeros((len(self.data_x), 4))

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


class Dataset_Genesis(Dataset):
    """
    Genesis Dataset
    Fixed borders: Train(0~4055), Val(4056~6000), Test(6001~end)
    Columns: 18 features, Label
    Label column is excluded from features.
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv',
                 target='MotorData.ActCurrent', scale=True, timeenc=0, freq='h'):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

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
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Sensor columns (exclude Label)
        sensor_cols = [c for c in df_raw.columns if c != 'Label']
        print(f"[Genesis-{self.flag.upper()}] Sensor cols: {sensor_cols[0]} ... {sensor_cols[-1]} (Total: {len(sensor_cols)})")
        print(f"[Genesis-{self.flag.upper()}] Total data length: {len(df_raw)}")

        # Fixed borders
        #   Train: 0 ~ 4055 (4056 rows)
        #   Val:   4056 ~ 6000 (1945 rows)
        #   Test:  6001 ~ end
        num_train = 4056
        num_val_end = 6001
        border1s = [0, num_train - self.seq_len, num_val_end - self.seq_len]
        border2s = [num_train, num_val_end, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        print(f"[Genesis-{self.flag.upper()}] Data range: [{border1}, {border2}) ({border2 - border1} rows)")

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[sensor_cols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Scale using train data only
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # Dummy time stamp
        self.data_stamp = np.zeros((len(self.data_x), 4))

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


class Dataset_Daphnet(Dataset):
    """
    Daphnet Dataset
    Fixed borders: Train(0~9693), Val(9694~15000), Test(15001~end)
    Columns: 9 features, Label
    Label column is excluded from features.
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='018_Daphnet_id_1_HumanActivity_tr_9693_1st_20732.csv',
                 target='ankle_horiz_fwd', scale=True, timeenc=0, freq='h'):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

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
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Sensor columns (exclude Label)
        sensor_cols = [c for c in df_raw.columns if c != 'Label']
        print(f"[Daphnet-{self.flag.upper()}] Sensor cols: {sensor_cols[0]} ... {sensor_cols[-1]} (Total: {len(sensor_cols)})")
        print(f"[Daphnet-{self.flag.upper()}] Total data length: {len(df_raw)}")

        # Fixed borders
        #   Train: 0 ~ 9693 (9694 rows)
        #   Val:   9694 ~ 15000 (5307 rows)
        #   Test:  15001 ~ end
        num_train = 9694
        num_val_end = 15001
        border1s = [0, num_train - self.seq_len, num_val_end - self.seq_len]
        border2s = [num_train, num_val_end, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        print(f"[Daphnet-{self.flag.upper()}] Data range: [{border1}, {border2}) ({border2 - border1} rows)")

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[sensor_cols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Scale using train data only
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # Dummy time stamp
        self.data_stamp = np.zeros((len(self.data_x), 4))

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


class Dataset_SWaT1(Dataset):
    """
    SWaT_id_1 Dataset
    Fixed borders: Train(0~3749), Val(3750~5500), Test(5501~end)
    Columns: 66 features, Label
    Label column is excluded from features.
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='171_SWaT_id_1_Sensor_tr_3749_1st_9522.csv',
                 target='value', scale=True, timeenc=0, freq='h'):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

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
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Sensor columns (exclude Label)
        sensor_cols = [c for c in df_raw.columns if c != 'Label']
        print(f"[SWaT1-{self.flag.upper()}] Sensor cols: {sensor_cols[0]} ... {sensor_cols[-1]} (Total: {len(sensor_cols)})")
        print(f"[SWaT1-{self.flag.upper()}] Total data length: {len(df_raw)}")

        # Fixed borders
        #   Train: 0 ~ 3749 (3750 rows)
        #   Val:   3750 ~ 5500 (1751 rows)
        #   Test:  5501 ~ end
        num_train = 3750
        num_val_end = 5501
        border1s = [0, num_train - self.seq_len, num_val_end - self.seq_len]
        border2s = [num_train, num_val_end, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        print(f"[SWaT1-{self.flag.upper()}] Data range: [{border1}, {border2}) ({border2 - border1} rows)")

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[sensor_cols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Scale using train data only
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # Dummy time stamp
        self.data_stamp = np.zeros((len(self.data_x), 4))

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


class Dataset_SWaT2(Dataset):
    """
    SWaT_id_2 Dataset
    Fixed borders: Train(0~23700), Val(23701~33700), Test(33701~end)
    Columns: 51 features, Label
    Label column is excluded from features.
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='172_SWaT_id_2_Sensor_tr_23700_1st_23800.csv',
                 target='FIT101', scale=True, timeenc=0, freq='h'):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

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
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Sensor columns (exclude Label)
        sensor_cols = [c for c in df_raw.columns if c != 'Label']
        print(f"[SWaT2-{self.flag.upper()}] Sensor cols: {sensor_cols[0]} ... {sensor_cols[-1]} (Total: {len(sensor_cols)})")
        print(f"[SWaT2-{self.flag.upper()}] Total data length: {len(df_raw)}")

        # Fixed borders
        #   Train: 0 ~ 23700 (23701 rows)
        #   Val:   23701 ~ 33700 (10000 rows)
        #   Test:  33701 ~ end
        num_train = 23701
        num_val_end = 33701
        border1s = [0, num_train - self.seq_len, num_val_end - self.seq_len]
        border2s = [num_train, num_val_end, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        print(f"[SWaT2-{self.flag.upper()}] Data range: [{border1}, {border2}) ({border2 - border1} rows)")

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[sensor_cols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Scale using train data only
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # Dummy time stamp
        self.data_stamp = np.zeros((len(self.data_x), 4))

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


class Dataset_CreditCard(Dataset):
    """
    CreditCard Dataset
    Fixed borders: Train(0~500), Val(501~1500), Test(1501~end)
    Columns: 29 features, Label
    Label column is excluded from features.
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='137_CreditCard_id_1_Finance_tr_500_1st_541.csv',
                 target='V1', scale=True, timeenc=0, freq='h'):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

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
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Sensor columns (exclude Label)
        sensor_cols = [c for c in df_raw.columns if c != 'Label']
        print(f"[CreditCard-{self.flag.upper()}] Sensor cols: {sensor_cols[0]} ... {sensor_cols[-1]} (Total: {len(sensor_cols)})")
        print(f"[CreditCard-{self.flag.upper()}] Total data length: {len(df_raw)}")

        # Fixed borders: Train(0~4920), Val(4921~5920), Test(5921~end)
        num_train = 4921
        num_val_end = 5921
        border1s = [0, num_train - self.seq_len, num_val_end - self.seq_len]
        border2s = [num_train, num_val_end, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        print(f"[CreditCard-{self.flag.upper()}] Data range: [{border1}, {border2}) ({border2 - border1} rows)")

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw[sensor_cols]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Scale using train data only
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # Dummy time stamp
        self.data_stamp = np.zeros((len(self.data_x), 4))

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
