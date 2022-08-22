from sklearn import metrics
from scipy import signal
from scipy.fft import fftshift

def auc_group(data=None, x=None, y=None):
    if x is None:
        x = data.index.astype('int64')
        y = data.values
    return metrics.auc(x, y)

def sum_abs_diffs(x):
    return np.sum(np.abs(np.diff(x)))

def calculate_features(df, time_index='timedelta', time_interval='5s', columns=None):
    df = df.assign(window_id=df.groupby(pd.Grouper(key=time_index, freq=time_interval)).ngroup())
    
    if columns is None:
        columns = list(map(lambda x: 'a' + str(x), range(7)))
    
    grouped_data = df.groupby('window_id')
    
    keystroke_features = list(map(lambda g: calculate_keystroke_features(g[1], columns=feature_cols), grouped_data))
    keystroke_features = pd.concat(keystroke_features)
    keystroke_features_grouped = keystroke_features.groupby('window_id').mean()
    keystroke_features_grouped = keystroke_features_grouped.reset_index()
    
    statistical_features = calculate_statistical_features(grouped_data[columns])
    statistical_features = statistical_features.reset_index()
    frequency_features = calculate_freq_features(df, columns)
    
    statistical_features = pd.merge_asof(statistical_features, keystroke_features_grouped, on=['window_id'], direction='nearest')
        
    all_features = pd.merge_asof(statistical_features, frequency_features, on=['window_id'], direction='nearest')

    return all_features

def calculate_statistical_features(grouped_data):
    # max
    features = grouped_data.max()
    features = features.add_prefix('max_')
    
    # min
    min_ = grouped_data.min()
    min_ = min_.add_prefix('min_')
    features = features.join(min_)
    
    # auc
    auc = grouped_data.agg(auc_group)
    auc = auc.add_prefix('auc_')
    features = features.join(auc)
    
    # var
    var_ = grouped_data.var()
    var_ = var_.add_prefix('var_')
    features = features.join(var_)
    
    # sum_diffs
    sum_diffs = grouped_data.transform(sum_abs_diffs)
    sum_diffs = sum_diffs.add_prefix('sum_diffs_')
    features = features.join(sum_diffs)
    
    return features


def calculate_freq_features(df, columns, fs=30, window_size=1, window='hann'):
    features = pd.DataFrame()    
    
    for column in columns:
        res = calculate_freq_features_per_column_data(df[column], fs=30, window_size=1, window='hann')
        res = res.add_suffix('_' + column)
        features = res.join(features)
    
    features = features.reset_index()
    features = features.assign(window_id=features.groupby(pd.Grouper(key='timedelta', freq='1s')).ngroup())

    return features

def calculate_freq_features_per_column_data(data, fs=30, window_size=1, window='hann'):
    if window == 'hann':
        window = signal.get_window('hann', int(fs * window_size))
        
    results = {}

    f, t, Sxx = signal.spectrogram(data, 30, window=window)
    
    # max freqs
    results['freqs_max'] = f[np.argmax(Sxx, 0)]
    
    # max amplitudes
    results['freqs_max_amp'] = np.max(Sxx, 0)
    
    # min amplitudes
    results['freqs_min_amp'] = np.min(Sxx, 0)
    
    # variance
    results['freqs_var'] = np.var(Sxx, 0)
    
    # mean
    results['freqs_mean'] = np.mean(Sxx, 0)
    
    # peak count
    peaks = np.apply_along_axis(signal.find_peaks, 0, Sxx)
    peaks = peaks[0]
    peak_count = list(map(lambda x: len(x), peaks))
    results['freqs_peak_count'] = peak_count

    results = pd.DataFrame(results, index=pd.TimedeltaIndex(t, unit='s'))
    results.index.rename('timedelta', inplace=True)
    
    return results

def merge_features(statistical_features, frequency_features):
    index = statistical_features.index.get_indexer(frequency_features.index, 'nearest')
    features = statistical_features
    features.loc[index, frequency_features.columns] = frequency_features
    return features