import time
import numpy as np
import pickle
import json
from datetime import datetime

offenseMap = {'BURGLARY': 0, 'ROBBERY': 1, 'GRAND LARCENY': 2, 'FELONY ASSAULT': 3}
offenseSet = set()
latSet = set()
lonSet = set()
timeSet = set()
data = []
raw_offense_counts = {k: 0 for k in offenseMap.keys()}
with open('Datasets/NYC_crime_1km/NYC_Crime.csv', 'r') as fs:
	fs.readline()
	for line in fs:
		arr = line.strip().split(',')
		print(arr)

		timeArray = time.strptime(arr[0], '%m/%d/%Y %I:%M:%S %p')
		timestamp = time.mktime(timeArray)
		offense = offenseMap[arr[2]]
		raw_offense_counts[arr[2]] += 1
		lat = float(arr[5])
		lon = float(arr[6])

		latSet.add(lat)
		lonSet.add(lon)
		timeSet.add(timestamp)
		offenseSet.add(offense)

		data.append({
			'time': timestamp,
			'offense': offense,
			'lat': lat,
			'lon': lon
		})
print('Length of data', len(data), '\n')
print('Offense:', offenseSet, '\n')
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
latDiv = 111 / 1
lonDiv = 84 / 1
latNum = int((maxLat - minLat) * latDiv) + 1
lonNum = int((maxLon - minLon) * lonDiv) + 1
trnTensor = np.zeros((latNum, lonNum, 638, len(offenseSet)))
valTensor = np.zeros((latNum, lonNum, 31, len(offenseSet)))
tstTensor = np.zeros((latNum, lonNum, 61, len(offenseSet)))
split_event_counts = {'train': 0, 'val': 0, 'test': 0}

split_class_counts = {
    'train': np.zeros(len(offenseSet)),
    'val': np.zeros(len(offenseSet)),
    'test': np.zeros(len(offenseSet)),
}
for i in range(len(data)):
	tup = data[i]
	temT = time.localtime(tup['time'])

	if temT.tm_year == 2014:
		day = temT.tm_yday - 1
		tensor = trnTensor
		split_name = 'train'

	elif temT.tm_year == 2015 and temT.tm_mon < 10:
		day = 365 + temT.tm_yday - 1
		tensor = trnTensor
		split_name = 'train'

	elif temT.tm_year == 2015 and temT.tm_mon == 10:
		day = temT.tm_mday - 1
		tensor = valTensor
		split_name = 'val'

	elif temT.tm_year == 2015 and temT.tm_mon > 10:
		day = temT.tm_yday - 304 - 1
		tensor = tstTensor
		split_name = 'test'

	else:
		continue

	row = int((tup['lat'] - minLat) * latDiv)
	col = int((tup['lon'] - minLon) * lonDiv)
	offense = tup['offense']

	tensor[row][col][day][offense] += 1
	split_event_counts[split_name] += 1
	split_class_counts[split_name][offense] += 1

names = ['trn.pkl', 'val.pkl', 'tst.pkl']
tensors = [trnTensor, valTensor, tstTensor]
for i in range(len(names)):
	with open('Datasets/NYC_crime_1km/' + names[i], 'wb') as fs:
		pickle.dump(tensors[i], fs)

id_to_offense = {v: k for k, v in offenseMap.items()}

metadata = {

    "created_at": datetime.now().isoformat(),

    "source_csv": "NYC_Crime.csv",

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

with open('Datasets/NYC_crime_1km/metadata.json', 'w') as fs:
    json.dump(metadata, fs, indent=4)

with open('Datasets/NYC_crime_1km/metadata.pkl', 'wb') as fs:
    pickle.dump(metadata, fs)
