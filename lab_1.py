import pandas as pd
import numpy as np

data = pd.read_csv('Data_1.csv', sep=';')
data.columns = data.columns.str.strip()

if 'Station' in data.index.names:
    data.reset_index(inplace=True)

data = data.reset_index(drop=True)

def calculate_derivatives(group):
    depths = group['Depth [m]'].values
    temperatures = group['Temperature [degrees_C]'].values

    forward_diff = np.diff(temperatures) / np.diff(depths)
    backward_diff = np.concatenate(([np.nan], np.diff(temperatures) / np.diff(depths)))
    central_diff = np.concatenate(([np.nan], np.diff(temperatures) / np.diff(depths), [np.nan]))
    forward_diff_full = np.concatenate(([np.nan], forward_diff))

    while len(forward_diff_full) < len(group):
        forward_diff_full = np.concatenate((forward_diff_full, [np.nan]))

    while len(backward_diff) < len(group):
        backward_diff = np.concatenate((backward_diff, [np.nan]))

    group['Forward_Derivative'] = forward_diff_full[:len(group)]
    group['Backward_Derivative'] = backward_diff[:len(group)]
    group['Central_Derivative'] = central_diff[:len(group)]

    return group

data = data.groupby('Station', group_keys=False, include_groups=False).apply(
    lambda x: calculate_derivatives(x[['Depth [m]', 'Temperature [degrees_C]']]))

def find_thermocline(group):
    max_derivative = group['Central_Derivative'].idxmax()
    depth_start = group.loc[max_derivative, 'Depth [m]']
    thickness = group['Depth [m]'].iloc[-1] - depth_start

    return pd.Series([group['Station'].iloc[0], depth_start, thickness])

thermoclines = data.groupby('Station', group_keys=False, include_groups=False).apply(
    lambda x: find_thermocline(x[['Depth [m]', 'Central_Derivative', 'Station']]))

thermoclines.columns = ['Station', 'Depth_Start', 'Thickness']

thermoclines.to_csv('thermoclines.csv', index=False)
