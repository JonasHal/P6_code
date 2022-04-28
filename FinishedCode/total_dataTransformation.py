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

        periods = pd.to_datetime(self.end) + np.timedelta64(10, 'D') - pd.to_datetime(self.start)
        index_values = (pd.date_range(pd.to_datetime(self.start) - pd.Timedelta(2, 'D'), periods=periods.days*24, freq='H'))
        data_hourly = pd.DataFrame(index=index_values, columns=['total_kWhDelivered', 'carsCharging', 'carsIdle']).fillna(0)

        print("Counting Cars...")
        #Creates the dataframe of how much the total load is, how many cars are charging and how many cars are idle
        for i in range(len(self.data)):
            # For each charging session, calculate kWhDelivered pr hour.
            charging_per_hour = self.data.loc[i, 'kWhDelivered']/self.data.loc[i, 'chargingTime']
            # Find what time the charging begun
            connection_time = self.data.loc[i, 'connectionTime'].round(freq='H').tz_localize(None)
            # count 1 if the car is charging for each index of the charging hours
            for j in range(self.data.loc[i, 'chargingTime']):
                data_hourly.loc[connection_time + pd.Timedelta(j, 'h'), ['total_kWhDelivered', 'carsCharging']] += [charging_per_hour, 1]
            # count 1 if the car is idle for each index of the idle hours
            for k in range(self.data.loc[i, 'idleTime']):
                data_hourly.loc[connection_time + pd.Timedelta(self.data.loc[i, 'chargingTime'] + k, 'h'), "carsIdle"] += 1

        real_start = pd.to_datetime(self.start) + np.timedelta64(1, 'D')

        data_hourly.loc[:, ("Weekday")] = data_hourly.index.day_of_week

        #Remove eccess data
        data_hourly = data_hourly[(data_hourly.index >= real_start) & (data_hourly.index < self.end)]
        return data_hourly

    def remove_outliers(self):
        q1, q3 = np.quantile(self.data['chargingTime'], [0.25, 0.75])
        self.data = self.data[self.data['chargingTime'] < q3 + 1.5 * (q3 - q1)]
        return self


if __name__ == '__main__':
    start, end = "2018-10-01", "2018-11-01"
    df = ImportEV().getBoth(start_date=start, end_date=end)
    Total_df = createTotal(df, start, end).remove_outliers().getTotalData()
    print(Total_df.to_string())