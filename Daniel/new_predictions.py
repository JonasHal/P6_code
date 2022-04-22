from P6_code.FinishedCode.dataTransformation import ImportEV
import pandas as pd
import numpy as np

start, end = '2018-07-01', '2018-11-01'

caltech = ImportEV().getCaltech(start, end).reset_index(drop=True)
jpl = ImportEV().getJPL(start, end).reset_index(drop=True)

caltech["chargingTime"] = np.floor((pd.to_datetime(caltech["doneChargingTime"]) - pd.to_datetime(caltech["connectionTime"])) / np.timedelta64(1, 'h')).astype('int')
caltech["chargingTime"].replace(0, 1, inplace=True)

print(caltech.head().to_string())

periods = pd.to_datetime(end) + np.timedelta64(3, 'D') - pd.to_datetime(start)
index_values = (pd.date_range(start, periods=periods.days*24, freq='H'))

caltech_kWh = pd.DataFrame(index=index_values, columns=['total_kWhDelivered']).fillna(0)

print(caltech_kWh.index)

for i in range(len(caltech)):
    charging_per_hour = caltech.loc[i, 'kWhDelivered']/caltech.loc[i, 'chargingTime']
    connection_time = caltech.loc[i, 'connectionTime'].round(freq='H')
    for j in range(caltech.loc[i, 'chargingTime']):
        caltech_kWh.loc[connection_time + np.timedelta64(j, 'h'), 'total_kWhDelivered'] += charging_per_hour

caltech_kWh = caltech_kWh[caltech_kWh.index < end]
print(caltech_kWh.to_string())