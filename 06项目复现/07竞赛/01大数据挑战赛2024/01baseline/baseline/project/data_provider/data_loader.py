import os
import numpy as np
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')

class Dataset_Meteorology(Dataset):
    def __init__(self, root_path, data_path, size=None, features='MS'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.stations_num = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        data = np.load(os.path.join(self.root_path, self.data_path)) # (T, S, 1)    
        data = np.squeeze(data) # (T S)
        era5 = np.load(os.path.join(self.root_path, 'global_data.npy')) 
        repeat_era5 = np.repeat(era5, 3, axis=0)[:len(data), :, :, :] # (T, 4, 9, S)
        repeat_era5 = repeat_era5.reshape(repeat_era5.shape[0], -1, repeat_era5.shape[3]) # (T, 36, S)

        self.data_x = data
        self.data_y = data
        self.covariate = repeat_era5

    def __getitem__(self, index):
        station_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, station_id:station_id+1]
        seq_y = self.data_y[r_begin:r_end, station_id:station_id+1] # (L 1)
        t1 = self.covariate[s_begin:s_end, :, station_id:station_id+1].squeeze()
        t2 = self.covariate[r_begin:r_end, :, station_id:station_id+1].squeeze()
        seq_x = np.concatenate([t1, seq_x], axis=1) # (L 37)
        seq_y = np.concatenate([t2, seq_y], axis=1) # (L 37)
        return seq_x, seq_y

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.stations_num