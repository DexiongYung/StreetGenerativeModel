from pandas import DataFrame
from torch.utils.data import Dataset


class StreetDataset(Dataset):
    def __init__(self, df: DataFrame):
        df = df.dropna(subset=['ST_NAME', 'ZIP_CO'])
        self.data_frame = df[['ST_NAME', 'ZIP_CO']]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        return str(self.data_frame['ST_NAME'].iloc[index]), int(self.data_frame['ZIP_CO'].iloc[index])
