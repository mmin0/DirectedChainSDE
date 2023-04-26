
"""
-----------------------------
data_loading.py
(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
"""

## Necessary Packages
import numpy as np
import pathlib
import torch
import pandas as pd
import os

def MinMaxScaler(data):
    """Min Max normalizer.
  
    Args:
        - data: original data
  
    Returns:
        - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return np.nan_to_num(norm_data)
    


def stock_data_loading(seq_len, batch_size, device, T=1):
    """Load and preprocess real-world datasets.
  
    Args:
        - seq_len: sequence length
    
    Returns:
        - data: preprocessed data.
    """  
    _here = pathlib.Path(__file__).resolve().parent
    ts = torch.linspace(0, T, seq_len, device=device)
    ori_data = np.loadtxt(_here/"data"/"stock_data.csv", delimiter = ",",skiprows = 1)
        
    # Flip the data to make chronological data
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data = MinMaxScaler(ori_data)
    
    # Preprocess the dataset
    temp_data = []    
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)
    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))    
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    data = torch.tensor(data, dtype=torch.float).to(device)
    data_size = data.shape[-1]
    
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return ts, data_size, dataloader


def electric_data_loading_single(filename, batch_size, device, T=1):
    """Load and preprocess real-world datasets.
  
    Args:
        - seq_len: sequence length
    
    Returns:
        - data: preprocessed data.
    """  
    
    _here = pathlib.Path(__file__).resolve().parent
    
    oneline = np.loadtxt(_here/"data"/"electric"/filename, delimiter = ",",skiprows = 3, max_rows=1, dtype='str')
    ori_data = np.loadtxt(_here/"data"/"electric"/filename, delimiter = ",",skiprows = 3, usecols=range(2, len(oneline)))

    # Flip the data to make chronological data
    # Normalize the data
    ori_data = MinMaxScaler(np.nan_to_num(ori_data))
    
    # shuffle the dataset
    # idx = np.random.permutation(len(ori_data))    
    # data = []
    # for i in range(len(ori_data)):
    #     data.append(ori_data[idx[i]])
    ori_data = torch.tensor(ori_data, dtype=torch.float).to(device).unsqueeze(-1)
    
    return ori_data

def electric_data_loading(seq_len, batch_size, device, T=1):
    """Load and preprocess real-world datasets.
  
    Args:
        - seq_len: sequence length
    
    Returns:
        - data: preprocessed data.
    """  
    
    _here = pathlib.Path(__file__).resolve().parent
    
    list_files = os.listdir(_here/"data"/"electric")
    
    ori_data = []
    print("All files: ", list_files)
    for _filename in list_files:
        if not _filename.endswith(".csv"):
            continue
        print("====Loading file: {}====".format(_filename))
        ori_data.append(electric_data_loading_single(_filename, batch_size, device))
        print("====Loading file finished, data size={}".format(ori_data[-1].size()))
    ori_data = torch.cat(ori_data, dim=-1)
    data = []
    idx = np.random.permutation(len(ori_data))    
    for i in range(len(ori_data)):
        data.append(ori_data[idx[i]:idx[i]+1])
    
    data = torch.cat(data, dim=0)
    seq_len, data_size = data.shape[1], data.shape[2]
    
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    ts = torch.linspace(0, T, seq_len, device=device)
    return ts, data_size, dataloader


def energy_data_loading(seq_len, batch_size, device, T=1):
    """Load and preprocess real-world datasets.
  
    Args:
        - seq_len: sequence length
    
    Returns:
        - data: preprocessed data.
    """  
    _here = pathlib.Path(__file__).resolve().parent
    ts = torch.linspace(0, T, seq_len, device=device)
    ori_data = np.loadtxt(_here/"data"/"energy_data.csv", delimiter = ",",skiprows = 1)
        
    # Flip the data to make chronological data
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data = MinMaxScaler(ori_data)
    
    # Preprocess the dataset
    temp_data = []    
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)
    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))    
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    data = torch.tensor(data, dtype=torch.float).to(device)[..., :-2]
    data_size = data.shape[-1]
    
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    
    return ts, data_size, dataloader


class Pipeline:
    def __init__(self, steps):
        """ Pre- and postprocessing pipeline. """
        self.steps = steps

    def transform(self, x, until=None):
        x = x.clone()
        for n, step in self.steps:
            if n == until:
                break
            x = step.transform(x)
        return x

    def inverse_transform(self, x, until=None):
        for n, step in self.steps[::-1]:
            if n == until:
                break
            x = step.inverse_transform(x)
        return x


class StandardScalerTS():
    """ Standard scales a given (indexed) input vector along the specified axis. """

    def __init__(self, axis=(1)):
        self.mean = None
        self.std = None
        self.axis = axis

    def transform(self, x):
        if self.mean is None:
            self.mean = torch.mean(x, dim=self.axis)
            self.std = torch.std(x, dim=self.axis)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)


def get_equities_dataset(assets=('SPX', 'DJI'), with_vol=True):
    """
    Get different returns series.
    """
    _here = pathlib.Path(__file__).resolve().parent
    oxford = pd.read_csv(_here/"data"/"oxfordmanrealizedvolatilityindices.csv")

    start = '2005-01-01 00:00:00+01:00'
    end = '2020-01-01 00:00:00+01:00'

    if assets == ('SPX',):
        df_asset = oxford[oxford['Symbol'] == '.SPX'].set_index(['Unnamed: 0'])  # [start:end]
        price = np.log(df_asset[['close_price']].values)
        rtn = (price[1:] - price[:-1]).reshape(1, -1, 1)
        vol = np.log(df_asset[['medrv']].values[-rtn.shape[1]:]).reshape(1, -1, 1)
        data_raw = np.concatenate([rtn, vol], axis=-1)
    elif assets == ('SPX', 'DJI'):
        df_spx = oxford[oxford['Symbol'] == '.SPX'].set_index(['Unnamed: 0'])[start:end]
        df_dji = oxford[oxford['Symbol'] == '.DJI'].set_index(['Unnamed: 0'])[start:end]
        index = df_dji.index.intersection(df_spx.index)
        df_dji = df_dji.loc[index]
        df_spx = df_spx.loc[index]
        price_spx = np.log(df_spx[['close_price']].values)
        rtn_spx = (price_spx[1:] - price_spx[:-1]).reshape(1, -1, 1)
        vol_spx = np.log(df_spx[['medrv']].values).reshape(1, -1, 1)
        price_dji = np.log(df_dji[['close_price']].values)
        rtn_dji = (price_dji[1:] - price_dji[:-1]).reshape(1, -1, 1)
        vol_dji = np.log(df_dji[['medrv']].values).reshape(1, -1, 1)
        data_raw = np.concatenate([rtn_spx, vol_spx[:, 1:], rtn_dji, vol_dji[:, 1:]], axis=-1)
    else:
        raise NotImplementedError()
    data_raw = torch.from_numpy(data_raw).float()
    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    data_preprocessed = pipeline.transform(data_raw)
    return pipeline, data_raw, data_preprocessed

def equity_data_loading(seq_len, batch_size, device, T=1, **data_params):
    _, _, x = get_equities_dataset(**data_params)
    assert x.shape[0] == 1
    
    tensor = torch.cat([x[:, t:t+seq_len] for t in range(x.shape[1] - seq_len)]).to(device)
    dataset = torch.utils.data.TensorDataset(tensor)
    ts = torch.linspace(0, T, seq_len, device=device)
    
    return ts, x.shape[-1], torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)