import os
import pandas
import pathlib
import urllib.request
import zipfile
import torch
import torchcde

from . import common


_here = pathlib.Path(__file__).resolve().parent


def _download():
    raw_data_folder = _here / 'data' / 'air_quality'
    loc = raw_data_folder / 'air_quality.zipz'
    if os.path.exists(loc):
        return
    os.makedirs(raw_data_folder, exist_ok=True)
    urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip',
                               loc)
    with zipfile.ZipFile(loc, 'r') as f:
        f.extractall(str(raw_data_folder))


def _air_quality_data():
    _download()
    raw_data_folder = _here / 'data' / 'air_quality'

    X = []
    y = []
    y_index = -1
    for filename in os.listdir(raw_data_folder / 'PRSA_Data_20130301-20170228'):
        y_index += 1
        content = pandas.read_csv(raw_data_folder / 'PRSA_Data_20130301-20170228' / filename)
        last_day = 1
        Xi = []
        for row in content.itertuples():
            if row.day != last_day:
                if len(Xi) == 24:
                    X.append(Xi)
                    y.append(y_index)
                    last_day = row.day
                Xi = []
            Xi.append([row._6,
                       row.PM10,
                       row.SO2,
                       row.NO2,
                       row.CO,
                       row.O3])
        if len(Xi) == 24:
            X.append(Xi)
            y.append(y_index)
    X = torchcde.linear_interpolation_coeffs(torch.tensor(X, dtype=torch.float32))
    y = torch.nn.functional.one_hot(torch.tensor(y)).to(torch.float32)

    return X, y


    

def air_quality_data(batch_size):
    data_folder = _here / 'processed_data' / 'air_quality'
    file = data_folder / 'x.pt'
    if os.path.exists(file):
        X = torch.load(file)
        y = torch.load(data_folder / 'y.pt')
    else:
        os.makedirs(data_folder, exist_ok=True)
        X, y = _air_quality_data()
        std, mean = torch.std_mean(X)
        X = (X - mean) / (std + 1e-5)
        torch.save(X, file)
        torch.save(y, data_folder / 'y.pt')

    t = torch.linspace(0, X.size(1) - 1, X.size(1))
    
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = common.dataloader(dataset, batch_size=batch_size)
    input_channels = X.size(-1)
    label_channels = y.size(-1)
    return t, dataloader, input_channels, label_channels
