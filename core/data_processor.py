import math
import numpy as np
import pandas as pd

"""
Modified LSTM implementation for water debit prediction
Based on Jakob Aungiers' implementation
"""
import json
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Modified DataLoader class for water debit data
class DataLoader():
    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        dataframe['Date'] = pd.to_datetime(dataframe['Date'], format='%d/%m/%Y %H:%M:%S')
        dataframe = dataframe.sort_values('Date')
        
        # Resample data to regular intervals (e.g., hourly)
        # You can adjust the frequency as needed: 'H' for hourly, 'D' for daily, etc.
        dataframe = dataframe.set_index('Date')
        dataframe = dataframe.resample('D').mean().fillna(method='ffill')
        self.data = dataframe['Debit'].values
        i_split = int(len(self.data) * split)
        self.data_train = self.data[:i_split]
        self.data_test = self.data[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        '''Create x, y test data windows'''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        
        # Reshape for single feature
        data_windows = data_windows.reshape((data_windows.shape[0], data_windows.shape[1], 1))
        
        if normalise:
            data_windows = self.normalise_windows(data_windows, single_window=False)

        x = data_windows[:, :-1]
        y = data_windows[:, -1]
        return x, y

    def get_train_data(self, seq_len, normalise):
        '''Create x, y train data windows'''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Generator for training data'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generate next data window'''
        window = self.data_train[i:i+seq_len]
        window = window.reshape((seq_len, 1))
        if normalise:
            window = self.normalise_windows(window, single_window=True)[0]
        x = window[:-1]
        y = window[-1]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with handling for zero values'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                col_data = window[:, col_i]
                # Find the first non-zero value or use mean if all zeros
                base_value = next((x for x in col_data if x != 0), None)
                if base_value is None:
                    base_value = 1.0
                elif base_value == 0:
                    base_value = 0.1
                normalised_col = []
                for p in col_data:
                    if p == 0:
                        # Handle zero values specially
                        normalised_col.append(0.0)
                    else:
                        normalised_col.append((float(p) / float(base_value)) - 1)
                
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T
            normalised_data.append(normalised_window)
        return np.array(normalised_data)