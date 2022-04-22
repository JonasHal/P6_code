from P6_code.FinishedCode.dataTransformation import ImportEV
import pandas as pd
import numpy as np

start, end = '2018-07-01', '2018-11-01'

caltech = ImportEV().getCaltech(start, end).reset_index(drop=True)
jpl = ImportEV().getJPL(start, end).reset_index(drop=True)

caltech["chargingTime"] = np.floor((pd.to_datetime(caltech["doneChargingTime"]) - pd.to_datetime(caltech["connectionTime"])) / np.timedelta64(1, 'h')).astype('int')
caltech["chargingTime"].replace(0, 1, inplace=True)
caltech["idleTime"] = np.floor((pd.to_datetime(caltech["disconnectTime"]) - pd.to_datetime(caltech["doneChargingTime"])) / np.timedelta64(1, 'h')).astype('int')

periods = pd.to_datetime(end) + np.timedelta64(3, 'D') - pd.to_datetime(start)
index_values = (pd.date_range(start, periods=periods.days*24, freq='H'))

caltech_hourly = pd.DataFrame(index=index_values, columns=['total_kWhDelivered', 'carsCharging', 'carsIdle']).fillna(0)

for i in range(len(caltech)):
    charging_per_hour = caltech.loc[i, 'kWhDelivered']/caltech.loc[i, 'chargingTime']
    connection_time = caltech.loc[i, 'connectionTime'].round(freq='H').tz_localize(None)
    for j in range(caltech.loc[i, 'chargingTime']):
        caltech_hourly.loc[connection_time + pd.Timedelta(j, 'h'), ['total_kWhDelivered', 'carsCharging']] += [charging_per_hour, 1]
    for k in range(caltech.loc[i, 'idleTime']):
        caltech_hourly.loc[connection_time + pd.Timedelta(caltech.loc[i, 'chargingTime'] + k, 'h'), "carsIdle"] += 1

caltech_hourly = caltech_hourly[caltech_hourly.index < end]
print(caltech_hourly.to_string())