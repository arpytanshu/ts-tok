import numpy as np
from  scipy.stats import norm


def subsequence(ts, window_len):
    shape = (ts.size - window_len + 1, window_len)
    strides = ts.strides * 2
    return np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)


def reconstruct_timeseries(timestamps, values, time_delta, fill_method='zero', return_mask=False):
    '''
    Reconstructs a regularly sampled time-series from irregularly sampled time-series.
    Parameters
    ----------
    timestamps : np.array
        unix epoch timestamps of irregularly sampled time-series.
    values : np.array
        values of irregularly sampled time-series.
    time_delta : integer
        sampling frequency of time series in seconds.

    Returns
    -------
    reconstructed_ts : np.array
        timestamps of reconstructed regularly sampled time-series.
    reconstructed_ts : np.array
        values of reconstructed regularly sampled time-series.
    '''

    available_fill_methods = ['interpolate', 'zero', 'nan', 'custom']
    assert fill_method in available_fill_methods, \
        f'fill_method should be one of {available_fill_methods}'
        
    num_ts = ((timestamps[-1] - timestamps[0]) // time_delta) + 1    
    
    if num_ts > len(timestamps):
        reconstructed_ts = np.linspace(timestamps[0], timestamps[-1], num_ts, dtype=int) # backfilled timestamps 
        available_mask = np.in1d(reconstructed_ts, timestamps) # array of False, with True in place of timestamps provided by Engg.
        
        if fill_method == 'zero':
            reconstructed_values = np.zeros(len(reconstructed_ts), dtype=np.float32) # 0-array of length equal to backfilled timestamps  
            reconstructed_values[available_mask] = values # fill real values in 0-array     
        elif fill_method == 'interpolate':
            reconstructed_values = np.zeros(len(reconstructed_ts), dtype=np.float32)
            interp_values = np.interp((~available_mask).nonzero()[0], available_mask.nonzero()[0], values)
            reconstructed_values[available_mask] = values
            reconstructed_values[~available_mask] = interp_values
        else:
            reconstructed_values = np.zeros(len(reconstructed_ts), dtype=np.float32) * np.nan # NaN-array of length equal to backfilled timestamps  
            reconstructed_values[available_mask] = values # fill real values in 0-array 
            
    else:
        if return_mask:
            return timestamps, values, None
        else:
            return timestamps, values
        
    if return_mask:
        return reconstructed_ts, reconstructed_values, available_mask
    else:
        return reconstructed_ts, reconstructed_values


def _gen_daily_signal():
    num_days = 7
    num_dp_per_day = 24
    x = np.linspace(0, num_days*2*np.pi, num_dp_per_day*num_days)
    e = np.random.randn(num_dp_per_day*num_days) * 0.3
    x = np.sin(x+e) + 5
    return x

def _gen_monotonic_signal():
    num_days = 7
    num_dp_per_day = 24
    x = np.linspace(0, num_days*2*np.pi, num_dp_per_day*num_days)
    e = np.random.randn(num_dp_per_day*num_days) * 0.3
    x = x+e + 5
    return x