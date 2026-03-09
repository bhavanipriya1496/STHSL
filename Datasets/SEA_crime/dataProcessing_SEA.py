import time
import numpy as np
import pickle
import zipfile
import io
import csv

offenseMap = {'LARCENY-THEFT': 0, 'BURGLARY': 1, 'ASSAULT OFFENSES': 2, 'MOTOR VEHICLE THEFT': 3}
offenseSet = set()
latSet = set()
lonSet = set()
timeSet = set()
data = []

with zipfile.ZipFile('SEA_crime.zip', 'r') as zf:
	with zf.open('SEA_crime.csv') as f:
		fs = io.TextIOWrapper(f, encoding='utf-8')
		reader = csv.reader(fs)
		header = next(reader)
		print('Header:', header)
		for arr in reader:
			try:
				# print(arr)
				report_datetime = arr[1].strip() # colum index 1 is 'Report DateTime'
				offense_category = arr[6].strip().upper() # column index 6 is 'Offense Sub Category'
				lat_str = arr[9].strip() # column index 9 is 'Latitude'
				lon_str = arr[10].strip() # column index 10 is 'Longitude'

				if offense_category not in offenseMap:
					continue
				if lat_str in ['', 'REDACTED'] or lon_str in ['', 'REDACTED']:
					continue
                # Filter invalid / out-of-city coordinates
				if not (47.0 <= float(lat_str) <= 48.0 and -123.0 <= float(lon_str) <= -121.0):
					continue
			
				timeArray = time.strptime(report_datetime, '%Y %b %d %I:%M:%S %p')
				year = timeArray.tm_year
				if year < 2014 or year > 2026:
					continue
				timestamp = time.mktime(timeArray)
				offense = offenseMap[offense_category]
				lat = float(lat_str)
				lon = float(lon_str)

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
			except Exception as e:
				print('Skipped row:', arr)
				print('Reason:', e)
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
latDiv = 111 / 3 # for 3km grid
lonDiv = 84 / 3 # for 3km grid
latNum = int((maxLat - minLat) * latDiv) + 1
lonNum = int((maxLon - minLon) * lonDiv) + 1
num_regions = latNum * lonNum
occupied = set()
for tup in data:
	row = int((tup['lat'] - minLat) * latDiv)
	col = int((tup['lon'] - minLon) * lonDiv)
	occupied.add((row, col))

print('Grid shape:', latNum, 'x', lonNum)
print('Total spatial regions:', num_regions)
print('Occupied spatial regions:', len(occupied))

# Normalize min/max timestamps to day boundaries
minDayStruct = time.localtime(minTime)
maxDayStruct = time.localtime(maxTime)

start_day = time.strptime(f'{minDayStruct.tm_year}-{minDayStruct.tm_yday}', '%Y-%j')
end_day = time.strptime(f'{maxDayStruct.tm_year}-{maxDayStruct.tm_yday}', '%Y-%j')

start_ts = time.mktime(start_day)
end_ts = time.mktime(end_day)

total_days = int((end_ts - start_ts) / 86400) + 1 # 24 × 60 × 60 = 86400 seconds

# Relative split based on SEA dataset:
# (train+val) : test = 7 : 1
# validation = last 30 days of the train+val block
tst_days = total_days // 8
trn_total_days = total_days - tst_days
val_days = 30
trn_days = trn_total_days - val_days

if trn_days <= 0:
    raise ValueError(
        f'Not enough days in dataset for requested split. '
        f'total_days={total_days}, tst_days={tst_days}, val_days={val_days}, trn_days={trn_days}'
    )

print('Total days:', total_days)
print('Train days:', trn_days)
print('Validation days:', val_days)
print('Test days:', tst_days)

trnTensor = np.zeros((latNum, lonNum, trn_days, len(offenseSet)))
valTensor = np.zeros((latNum, lonNum, val_days, len(offenseSet)))
tstTensor = np.zeros((latNum, lonNum, tst_days, len(offenseSet)))

for i in range(len(data)):
    tup = data[i]
    temT = time.localtime(tup['time'])

    cur_day_struct = time.strptime(f'{temT.tm_year}-{temT.tm_yday}', '%Y-%j')
    cur_day_ts = time.mktime(cur_day_struct)
    global_day = int((cur_day_ts - start_ts) / 86400)

    if global_day < trn_days:
        day = global_day
        tensor = trnTensor
    elif global_day < trn_days + val_days:
        day = global_day - trn_days
        tensor = valTensor
    else:
        day = global_day - trn_days - val_days
        if day >= tst_days:
            continue
        tensor = tstTensor

    row = int((tup['lat'] - minLat) * latDiv)
    col = int((tup['lon'] - minLon) * lonDiv)
    offense = tup['offense']
    tensor[row][col][day][offense] += 1

names = ['trn.pkl', 'val.pkl', 'tst.pkl']
tensors = [trnTensor, valTensor, tstTensor]
for i in range(len(names)):
    with open(names[i], 'wb') as fs:
        pickle.dump(tensors[i], fs)

print('Saved:')
print('Datasets/SEA_crime/trn.pkl', trnTensor.shape)
print('Datasets/SEA_crime/val.pkl', valTensor.shape)
print('Datasets/SEA_crime/tst.pkl', tstTensor.shape)

# -------------------------
# Save metadata for reference (paper / debugging)
# -------------------------

meta = {
	"dataset_name": "SEA",

	"source": {
		"zip_file": "Datasets/SEA_crime/SEA_crime.zip",
		"csv_file": "SEA_crime.csv"
	},

	"columns_used": {
		"datetime_column_index": 1,
		"datetime_column_name": "Report DateTime",
		"offense_column_index": 6,
		"offense_column_name": "Offense Sub Category",
		"latitude_column_index": 9,
		"longitude_column_index": 10
	},

	"offense_mapping": offenseMap,

	"spatial_grid": {
		"min_lat": minLat,
		"max_lat": maxLat,
		"min_lon": minLon,
		"max_lon": maxLon,
		"lat_division_factor": latDiv,
		"lon_division_factor": lonDiv,
		"grid_cell_size_km": 3,
		"lat_cells": latNum,
		"lon_cells": lonNum,
		"num_regions": latNum * lonNum,
		"occupied_regions": len(occupied)
	},

	"time_range": {
		"min_timestamp": minTime,
		"max_timestamp": maxTime,
		"start_day_timestamp": start_ts,
		"end_day_timestamp": end_ts,
		"total_days": total_days,
		"start_date": time.strftime('%Y-%m-%d', time.localtime(start_ts)),
		"end_date": time.strftime('%Y-%m-%d', time.localtime(end_ts))
	},

	"dataset_split": {
		"rule": "(train+val):test = 7:1, validation = last 30 days of train block",
		"train_days": trn_days,
		"val_days": val_days,
		"test_days": tst_days
	},

	"tensor_shapes": {
		"train_tensor": trnTensor.shape,
		"val_tensor": valTensor.shape,
		"test_tensor": tstTensor.shape,
		"axis_order": ["latitude_index", "longitude_index", "day_index", "offense_category"]
	},

	"notes": {
		"lat_conversion": "1 degree latitude ≈ 111 km",
		"lon_conversion": "1 degree longitude ≈ 84 km (approx used for consistency with STHSL)",
		"time_unit": "timestamps stored in Unix seconds",
		"grid_resolution": "3 km × 3 km cells"
	}
}

meta_path = "meta.pkl"

with open(meta_path, "wb") as f:
	pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

print("[INFO] Saved metadata file:", meta_path)