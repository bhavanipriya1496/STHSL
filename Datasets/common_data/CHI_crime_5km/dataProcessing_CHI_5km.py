import csv
import json
import pickle
import time
from datetime import datetime
import numpy as np

offenseMap = {'ASSAULT': 0, 'BATTERY': 1, 'CRIMINAL DAMAGE': 2, 'THEFT': 3}

offenseSet = set()
latSet = set()
lonSet = set()
timeSet = set()
data = []
raw_offense_counts = {k: 0 for k in offenseMap.keys()}

# Read and filter source CSV
with open('CHI_Crime.csv', 'r', newline='', encoding='utf-8') as fs:
    reader = csv.reader(fs)
    next(reader)  # skip header

    for row_idx, arr in enumerate(reader, start=2):
        datetime_str = arr[0].strip()

        if not datetime_str:
            continue

        try:
            timeArray = time.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S.%f')
            timestamp = time.mktime(timeArray)
        except ValueError:
            continue

        offense_name = arr[1].strip()
        if offense_name not in offenseMap:
            continue

        try:
            lat = float(arr[2].strip())
            lon = float(arr[3].strip())
        except ValueError:
            continue

        offense = offenseMap[offense_name]
        raw_offense_counts[offense_name] += 1

        offenseSet.add(offense_name)
        latSet.add(lat)
        lonSet.add(lon)
        timeSet.add(timestamp)

        data.append({
            'time': timestamp,
            'lat': lat,
            'lon': lon,
            'offense': offense
        })

print('Length of data', len(data), '\n')
print('Offense:', offenseSet, '\n')

if len(data) == 0:
    raise ValueError("No valid events found after filtering. Check offense names and CSV column mapping.")

if len(latSet) == 0 or len(lonSet) == 0 or len(timeSet) == 0:
    raise ValueError("Latitude/Longitude/Time sets are empty. Parsing failed.")

print('Latitude:', min(latSet), max(latSet))
print('Longtitude:', min(lonSet), max(lonSet))
print('Latitude:', min(latSet), max(latSet), (max(latSet) - min(latSet)) / (1 / 111), '\n')
print('Longtitude:', min(lonSet), max(lonSet), (max(lonSet) - min(lonSet)) / (1 / 84), '\n')
print('Time:')

minTime = min(timeSet)
maxTime = max(timeSet)
print(time.localtime(minTime))
print(time.localtime(maxTime))

minLat = min(latSet)
minLon = min(lonSet)
maxLat = max(latSet)
maxLon = max(lonSet)

latDiv = 111 / 5  # 5 km resolution grid
lonDiv = 84 / 5   # 5 km resolution grid

latNum = int((maxLat - minLat) * latDiv) + 1
lonNum = int((maxLon - minLon) * lonDiv) + 1

num_classes = len(offenseMap)

trnTensor = np.zeros((latNum, lonNum, 547, num_classes))
valTensor = np.zeros((latNum, lonNum, 31, num_classes))
tstTensor = np.zeros((latNum, lonNum, 153, num_classes))

split_event_counts = {'train': 0, 'val': 0, 'test': 0}

split_class_counts = {
    'train': np.zeros(num_classes),
    'val': np.zeros(num_classes),
    'test': np.zeros(num_classes),
}

for i in range(len(data)):
    tup = data[i]
    temT = time.localtime(tup['time'])

    if temT.tm_year == 2015:
        day = temT.tm_yday - 1
        tensor = trnTensor
        split_name = 'train'

    elif temT.tm_year == 2016 and temT.tm_mon <= 6:
        day = 365 + temT.tm_yday - 1
        tensor = trnTensor
        split_name = 'train'

    elif temT.tm_year == 2016 and temT.tm_mon == 7:
        day = temT.tm_mday - 1
        tensor = valTensor
        split_name = 'val'

    elif temT.tm_year == 2016 and temT.tm_mon >= 8:
        # Aug 1 is day 214 in leap year 2016
        day = temT.tm_yday - 214
        tensor = tstTensor
        split_name = 'test'

    else:
        continue

    row = int((tup['lat'] - minLat) * latDiv)
    col = int((tup['lon'] - minLon) * lonDiv)
    offense = tup['offense']

    if row < 0 or row >= latNum or col < 0 or col >= lonNum:
        continue

    tensor[row][col][day][offense] += 1
    split_event_counts[split_name] += 1
    split_class_counts[split_name][offense] += 1

names = ['trn.pkl', 'val.pkl', 'tst.pkl']
tensors = [trnTensor, valTensor, tstTensor]

for i in range(len(names)):
    with open(names[i], 'wb') as fs:
        pickle.dump(tensors[i], fs)

id_to_offense = {v: k for k, v in offenseMap.items()}

metadata = {

    "created_at": datetime.now().isoformat(),

    "source_csv": "CHI_Crime.csv",

    "offense_map": offenseMap,
    "id_to_offense": id_to_offense,
    "raw_offense_counts": raw_offense_counts,

    "num_events": len(data),

    "time_range": {
        "min": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(minTime)),
        "max": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(maxTime)),
    },

    "spatial_bounds": {
        "min_lat": minLat,
        "max_lat": maxLat,
        "min_lon": minLon,
        "max_lon": maxLon,
    },

    "grid": {
        "lat_cells": latNum,
        "lon_cells": lonNum,
        "cell_km": 5,
    },

    "tensor_shapes": {
        "train": list(trnTensor.shape),
        "val": list(valTensor.shape),
        "test": list(tstTensor.shape),
    },

    "split_event_counts": split_event_counts,

    "split_class_counts": {
        "train": split_class_counts["train"].tolist(),
        "val": split_class_counts["val"].tolist(),
        "test": split_class_counts["test"].tolist(),
    }

}


total_regions = latNum * lonNum

occupied_total = np.sum(
    (np.sum(trnTensor, axis=(2, 3)) +
     np.sum(valTensor, axis=(2, 3)) +
     np.sum(tstTensor, axis=(2, 3))) > 0
)

occupied_train = np.sum(np.sum(trnTensor, axis=(2, 3)) > 0)
occupied_val = np.sum(np.sum(valTensor, axis=(2, 3)) > 0)
occupied_test = np.sum(np.sum(tstTensor, axis=(2, 3)) > 0)

lat_span_km = (maxLat - minLat) * 111
lon_span_km = (maxLon - minLon) * 84

minTimeStr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(minTime))
maxTimeStr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(maxTime))

print("\n========== DATASET SUMMARY ==========")

print("\nTime span:")
print("Start:", minTimeStr)
print("End  :", maxTimeStr)

print("\nTotal events:", len(data))

print("\nCrime types:", offenseMap)

print("\nRaw offense counts:")
for k, v in raw_offense_counts.items():
    print(k, ":", v)

print("\nSpatial bounds:")
print("Latitude :", minLat, "to", maxLat)
print("Longitude:", minLon, "to", maxLon)

print("\nSpatial span (approx km):")
print("Lat span:", lat_span_km)
print("Lon span:", lon_span_km)

print("\nGrid info:")
print("Lat cells:", latNum)
print("Lon cells:", lonNum)
print("Total regions:", total_regions)

print("\nOccupied regions:")
print("Total :", int(occupied_total))
print("Train :", int(occupied_train))
print("Val   :", int(occupied_val))
print("Test  :", int(occupied_test))

print("\nTensor shapes:")
print("Train:", trnTensor.shape)
print("Val  :", valTensor.shape)
print("Test :", tstTensor.shape)

print("\nSplit event counts:")
print(split_event_counts)

print("\nSplit class counts:")
print("Train:", split_class_counts["train"])
print("Val  :", split_class_counts["val"])
print("Test :", split_class_counts["test"])

print("\n====================================\n")

with open('metadata.json', 'w') as fs:
    json.dump(metadata, fs, indent=4)

with open('metadata.pkl', 'wb') as fs:
    pickle.dump(metadata, fs)
