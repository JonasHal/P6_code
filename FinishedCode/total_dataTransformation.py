from P6_code.FinishedCode.dataTransformation import ImportEV
import pandas as pd
import numpy as np


class createTotal:
    def __init__(self, dataframe, start, end):
        self.data = dataframe
        self.start = start
        self.end = end

    def getTotalData(self):
        self.data["chargingTime"] = np.floor((pd.to_datetime(self.data["doneChargingTime"]) - pd.to_datetime(self.data["connectionTime"])) / np.timedelta64(1, 'h')).astype('int')
        self.data["chargingTime"].replace(0, 1, inplace=True)
        self.data["idleTime"] = np.floor((pd.to_datetime(self.data["disconnectTime"]) - pd.to_datetime(self.data["doneChargingTime"])) / np.timedelta64(1, 'h')).astype('int')

        periods = pd.to_datetime(self.end) + np.timedelta64(3, 'D') - pd.to_datetime(self.start)
        index_values = (pd.date_range(self.start, periods=periods.days*24, freq='H'))

        data_hourly = pd.DataFrame(index=index_values, columns=['total_kWhDelivered', 'carsCharging', 'carsIdle']).fillna(0)

        print("Counting Cars...")
        for i in range(len(self.data)):
            charging_per_hour = self.data.loc[i, 'kWhDelivered']/self.data.loc[i, 'chargingTime']
            connection_time = self.data.loc[i, 'connectionTime'].round(freq='H').tz_localize(None)
            for j in range(self.data.loc[i, 'chargingTime']):
                data_hourly.loc[connection_time + pd.Timedelta(j, 'h'), ['total_kWhDelivered', 'carsCharging']] += [charging_per_hour, 1]
            for k in range(self.data.loc[i, 'idleTime']):
                data_hourly.loc[connection_time + pd.Timedelta(self.data.loc[i, 'chargingTime'] + k, 'h'), "carsIdle"] += 1

        data_hourly = data_hourly[data_hourly.index < self.end]
        return data_hourly


if __name__ == '__main__':
    start, end = "2018-05-01", "2018-11-01"
    df = ImportEV().getBoth(start_date=start, end_date=end)
    Total_df = createTotal(df, start, end).getTotalData()
    print(Total_df.to_string())