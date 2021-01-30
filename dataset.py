import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class kddcup99_Dataset(Dataset):
    def __init__(self, data_dir = None, mode='train',ratio = 0.8):
        self.mode = mode
        
        data = np.load(data_dir,allow_pickle=True)['kdd']
        data = np.float32(data)
        data = torch.from_numpy(data)
        
        
        normal_data = data[data[:, -1] == 0]
        abnormal_data = data[data[:, -1] == 1]

        train_normal_mark = int(normal_data.shape[0] * ratio)
        train_abnormal_mark = int(abnormal_data.shape[0] * ratio)

        train_normal_data = normal_data[:train_normal_mark, :]
        train_abnormal_data = abnormal_data[:train_abnormal_mark, :]
        self.train_data = np.concatenate((train_normal_data, train_abnormal_data), axis=0)
        np.random.shuffle(self.train_data)
        
        test_normal_data = normal_data[train_normal_mark:, :]
        test_abnormal_data = abnormal_data[train_abnormal_mark:, :]
        self.test_data = np.concatenate((test_normal_data, test_abnormal_data), axis=0)
        np.random.shuffle(self.test_data)
          
        
    def __len__(self):
        if self.mode == 'train':
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]
        
    def __getitem__(self, index):
        if self.mode == 'train':
            return self.train_data[index,:-1], self.train_data[index,-1]
        else:
            return self.test_data[index,:-1], self.test_data[index,-1]
        
        

def get_loader(hyp, mode = 'train'):
    """Build and return data loader."""

    dataset = kddcup99_Dataset(hyp['data_dir'], mode, hyp['ratio'])

    shuffle = True if mode == 'train' else False
    

    data_loader = DataLoader(dataset=dataset,
                             batch_size=hyp['batch_size'],
                             shuffle=shuffle)
    
    
    return data_loader,len(dataset)

