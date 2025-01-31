import numpy as np
import pandas as pd

def deg_to_radians(deg):
    return deg * (np.pi / 180)

def distance_between(lat1, lon1, lat2, lon2):
    delta_lat = deg_to_radians(lat2 - lat1)
    delta_lon = deg_to_radians(lon2 - lon1)

    a = (np.sin(delta_lat / 2) ** 2 + np.cos(deg_to_radians(lat1)) * np.cos(deg_to_radians(lat2)) * np.sin(
        delta_lon / 2) ** 2)
    c = 2 * np.atan2(np.sqrt(a), np.sqrt(1 - a))

    return 6371000 * c

def load_station_data(file_path):
    df = pd.read_csv(file_path, delimiter='\t')
    stations = []

    for _, row in df.iterrows():
        stations.append({
            'Station': row['Station'],
            'Longitude': float(row['Longitude']),
            'Latitude': float(row['Latitude']),
            'Depth': float(row['Depth']),
            'Ve': float(row['Ve']),
            'Vn': float(row['Vn'])
        })

    return stations

def integrate_left(func, start, end, intervals):
    step = (end - start) / intervals

    return step * sum(func(start + i * step) for i in range(intervals))

def integrate_right(func, start, end, intervals):
    step = (end - start) / intervals

    return step * sum(func(start + (i + 1) * step) for i in range(intervals))

def integrate_trapezoidal(func, start, end, intervals):
    step = (end - start) / intervals

    return step * (0.5 * (func(start) + func(end)) + sum(func(start + i * step) for i in range(1, intervals)))

def integrate_simpson(func, start, end, intervals):
    if intervals % 2 != 0:
        intervals += 1

    step = (end - start) / intervals

    return (step / 3) * (func(start) + func(end) + 4 * sum(func(start + i * step) for i in range(1, intervals, 2)) + 2 * sum(func(start + i * step) for i in range(2, intervals, 2)))

data = load_station_data('Data_3.txt')

integration_methods = {
    'left_rectangle': integrate_left,
    'right_rectangle': integrate_right,
    'trapezoidal': integrate_trapezoidal,
    'simpson': integrate_simpson
}

def value_at_depth(values, depths, x):
    if x in depths:
        return values[depths.index(x)]

    return 0

def compute_depth_integrals(station_data, method, component):
    aggregated_data = {}

    for entry in station_data:
        station = entry['Station']
        depth = entry['Depth']
        value = entry[component]

        if station not in aggregated_data:
            aggregated_data[station] = []

        aggregated_data[station].append((depth, value))

    results = {}

    for station, values in aggregated_data.items():
        values.sort(key=lambda x: x[0])
        depths = [v[0] for v in values]
        values = [v[1] for v in values]
        start = depths[0]
        end = depths[-1]
        n = len(values) - 1

        def func(x):
            return value_at_depth(values, depths, x)

        results[station] = method(func, start, end, n) if n > 0 else values[0]

    return results

def compute_station_integrals(station_data, depth_results, method):
    unique_stations = sorted(set(entry['Station'] for entry in station_data))
    lat_map = {entry['Station']: entry['Latitude'] for entry in station_data}
    lon_map = {entry['Station']: entry['Longitude'] for entry in station_data}

    distances = [0] + [distance_between(lat_map[unique_stations[i]], lon_map[unique_stations[i]], lat_map[unique_stations[i + 1]], lon_map[unique_stations[i + 1]])
        for i in range(len(unique_stations) - 1)]

    cumulative_distances = [sum(distances[:i + 1]) for i in range(len(distances))]

    results = []

    for comp in ['Ve', 'Vn']:
        values = [depth_results[comp][station] for station in unique_stations if station in depth_results[comp]]
        start = cumulative_distances[0]
        end = cumulative_distances[-1]
        n = len(values) - 1

        def func(x):
            return value_at_depth(values, cumulative_distances, x)

        results.append(method(func, start, end, n) if n > 0 else values[0])

    return results

for method_name, method in integration_methods.items():
    depth_results = {
        'Ve': compute_depth_integrals(data, method, 'Ve'),
        'Vn': compute_depth_integrals(data, method, 'Vn')
    }
    station_results = compute_station_integrals(data, depth_results, method)
    print(f"Method: {method_name}, Ve: {station_results[0]}, Vn: {station_results[1]}")
