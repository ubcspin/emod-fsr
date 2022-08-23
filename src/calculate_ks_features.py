import pandas as pd

from config import TIME_INDEX, TIME_INTERVAL


def calculate_keystroke_features_per_key(df, key='a1'):
    df = df.reset_index(drop=True)
    raw_key = key.split('_')[-1]

    start_index = pd.Index(df[raw_key + '_press_start']
                           ).get_indexer_for([True])
    end_index = pd.Index(df[raw_key + '_press_end']).get_indexer_for([True])

    if df[key].head(1).values != 0:
        start_index = np.insert(start_index, 0, df.index.values[0])
    if df[key].tail(1).values != 0:
        end_index = np.append(end_index, df.index.values[-1])

    column_names = [key + '_keystroke_duration',
                    key + '_keystroke_peak_count',
                    key + '_keystroke_max_peak',
                    key + '_keystroke_var',
                    key + '_keystroke_mean',
                    key + '_keystroke_auc',
                    key + '_keystroke_time_to_peak',
                    key + '_keystroke_peak_to_release',
                    ]
                    
    df[key + '_keystroke_duration'] = 0
    df[key + '_keystroke_peak_count'] = 0
    df[key + '_keystroke_max_peak'] = 0
    df[key + '_keystroke_var'] = 0
    df[key + '_keystroke_mean'] = 0
    df[key + '_keystroke_auc'] = 0
    df[key + '_keystroke_time_to_peak'] = 0
    df[key + '_keystroke_peak_to_release'] = 0
    error_count = 0

    for s, e in zip(start_index, end_index):
        df.loc[s:e, key + '_keystroke_duration'] = df.timestamps.iloc[e] - \
            df.timestamps.iloc[s]
        data = df.loc[s:e, ['timestamps', key]]
        timestamps = data.timestamps
        data = data[key]
        assert len(data) == len(timestamps)

        try:
            peaks, _ = signal.find_peaks(data)
            if not (peaks.size > 0):
                peak_max = np.max(data)
                peak_max_arg = np.argmax(data)
                peaks = (data > 0)
            else:
                peak_max = np.max(data[peaks])
                peak_max_arg = np.argmax(data)

            df.loc[s:e, key + '_keystroke_peak_count'] = sum(peaks)
            df.loc[s:e, key + '_keystroke_max_peak'] = peak_max
            df.loc[s:e, key + '_keystroke_time_to_peak'] = timestamps[peak_max_arg] - \
                df.timestamps.iloc[s]
            df.loc[s:e, key + '_keystroke_peak_to_release'] = df.timestamps.iloc[e] - \
                timestamps[peak_max_arg]
            df.loc[s:e, key + '_keystroke_var'] = np.var(data)
            df.loc[s:e, key + '_keystroke_mean'] = np.mean(data)
            df.loc[s:e, key +
                   '_keystroke_auc'] = auc_group(x=timestamps, y=data)

        except (ValueError, KeyError):
            error_count += 1

    df.loc[df[raw_key + '_down'], key + '_keystroke_duration'] = 0
    df = df.reset_index(drop=True)
    return df[column_names]


def calculate_keystroke_features(df, time_index=TIME_INDEX, time_interval=TIME_INTERVAL, columns=None):
    dfs = []
    if columns is None:
        columns = list(map(lambda x: 'a' + str(x), range(7)))
    for column in columns:
        dfs.append(calculate_keystroke_features_per_key(df, column))
    dfs = pd.concat(dfs, axis=1)
    try:
        dfs['window_id'] = df['window_id'].unique()[0]
    except KeyError:
        pass
    return dfs
