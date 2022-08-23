# TODO
from scipy import signal
from statsmodels.nonparametric import smoothers_lowess

def butter_lowpass(cutoff: float, fs: float, order=5):
    return signal.butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data: np.ndarray, cutoff=10, fs=30, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def filter_signals(df, cols):
    for col in cols:
        new_col = 'lpass_' + col
        df[new_col] = butter_lowpass_filter(df[col])
        df[new_col] = df[new_col].astype('int64')
    return df

def lowess(df, cols):
    for col in cols:
        df[col] = smoothers_lowess.lowess(df[col], df.timestamps, is_sorted=True, frac=0.025, it=0)
        df[col] = df[col].astype('int64')

    return df

def sgolay(df, cols):
    for col in cols:
        df[col] = signal.savgol_filter(df[col], 15, 10)
        df[col] = df[col].astype('int64')
    return df

def ewm(df, cols):
    for col in cols:
        new_col = 'smooth_' + col
        df[new_col] = df[col].ewm(alpha=0.05).mean().astype('int64')
    return df

# Performed as part of training pipeline:
# print('[INFO] Smoothing')
# df = ewm(df, lowpass_cols)
# df = filter_signals(df, lowpass_cols)