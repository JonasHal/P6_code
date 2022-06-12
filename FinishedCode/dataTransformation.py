import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np
from P6_code.FinishedCode.importData import ImportEV

# Data Transformation for each day
pd.options.mode.chained_assignment = None
# Preprocessing Functions
class createTransformation:
    """
    A class used to create the data transformations before applying Machine Learning
    """
    def __init__(self, dataframe, start, end):
        self.data = dataframe
        self.start = start
        self.end = end

    def remove_outliers(self):
        """
        Removes the chargingTime outliers from the dataframe
        """
        q1, q3 = np.quantile(self.data['chargingTime'], [0.25, 0.75])
        self.data = self.data[self.data['chargingTime'] < q3 + 1.5 * (q3 - q1)].reset_index(drop=True)
        return self

    def getUserData(self, user):
        """
        Creates a dataframe for one particular user. Each observation is the user charging time in one day.
        """
        #Slice the data to one user:
        user_df = self.data[self.data["userID"] == user]
        user_df = user_df[["connectionTime", "chargingTime", "kWhDelivered"]]

        # Outlier detection on charging time
        user_df = user_df[user_df["chargingTime"] > pd.Timedelta(0)]

        # Feature Information on charging days
        date_index_charging = pd.to_datetime(user_df.pop("connectionTime"))

        user_df.loc[:, ("connectionDay")] = pd.to_datetime(date_index_charging.dt.strftime('%Y-%m-%d'))
        user_df = user_df.groupby('connectionDay').agg({'chargingTime': np.sum, 'kWhDelivered': np.sum})

        # Index creation
        periods = pd.to_datetime(self.end) - pd.to_datetime(self.start)
        index_values = (pd.date_range(self.start, periods=periods.days, freq='D'))
        # Merge the days
        user_df = pd.merge(user_df, pd.DataFrame(data=pd.to_datetime(index_values), columns=["connectionDay"]),
                           how="outer", on="connectionDay")
        user_df = user_df.sort_values(by="connectionDay")
        # Additional Feature Informations
        date_index = pd.to_datetime(user_df.pop("connectionDay"))

        #Holiday Handling
        cal = calendar()
        holidays = cal.holidays(start=self.start, end=self.end)

        #Input Features on specific days
        user_df.loc[:, 'Holiday'] = date_index.isin(holidays)
        user_df.loc[:, ("Weekday")] = date_index.dt.day_of_week

        #Impute 0 on missing values
        user_df["chargingTime"].fillna(pd.Timedelta('0 days'), inplace=True)
        user_df["kWhDelivered"].fillna(0, inplace=True)

        #Timedelta to numeric on ConnectionTime
        user_df['chargingTime'] = user_df['chargingTime'] / np.timedelta64(1, 's')

        return user_df.reset_index(drop=True)

    def getTotalData(self):
        """
        Creates a dataframe for the total kWhDelivered, carsCharging and carsIdle. Each observation is one hour.
        """

        #Calculate how many hours the car were charging and idle
        self.data["chargingTime"] = np.floor((pd.to_datetime(self.data["doneChargingTime"]) - pd.to_datetime(self.data["connectionTime"])) / np.timedelta64(1, 'h')).astype('int')
        self.data["chargingTime"].replace(0, 1, inplace=True)
        self.data["idleTime"] = np.floor((pd.to_datetime(self.data["disconnectTime"]) - pd.to_datetime(self.data["doneChargingTime"])) / np.timedelta64(1, 'h')).astype('int')

        #Create the index values and empty dataframe
        periods = pd.to_datetime(self.end) + np.timedelta64(10, 'D') - pd.to_datetime(self.start)
        index_values = (pd.date_range(pd.to_datetime(self.start) - pd.Timedelta(2, 'D'), periods=periods.days*24, freq='H'))
        data_hourly = pd.DataFrame(index=index_values, columns=['total_kWhDelivered', 'carsCharging', 'carsIdle']).fillna(0)

        print("Counting Cars...")
        #Creates the dataframe of how much the total load is, how many cars are charging and how many cars are idle

        #Iterate though every charging session and add the kWhDelivered, carsCharging, carsIdle at the exact time of the session
        for i in range(len(self.data)):
            # For each charging session, calculate kWhDelivered pr hour.
            charging_per_hour = self.data.loc[i, 'kWhDelivered']/self.data.loc[i, 'chargingTime']
            # Find what time the charging begun
            connection_time = self.data.loc[i, 'connectionTime'].round(freq='H').tz_localize(None)

            # count += 1 if the car is charging for each index of the charging hours
            for j in range(self.data.loc[i, 'chargingTime']):
                data_hourly.loc[connection_time + pd.Timedelta(j, 'h'), ['total_kWhDelivered', 'carsCharging']] += [charging_per_hour, 1]

            # count += 1 if the car is idle for each index of the idle hours
            for k in range(self.data.loc[i, 'idleTime']):
                data_hourly.loc[connection_time + pd.Timedelta(self.data.loc[i, 'chargingTime'] + k, 'h'), "carsIdle"] += 1

        real_start = pd.to_datetime(self.start) + np.timedelta64(1, 'D')

        # Holiday Handling
        cal = calendar()
        holidays = cal.holidays(start=self.start, end=self.end)

        # Input Features on specific days
        data_hourly.loc[:, 'Holiday'] = data_hourly.index.isin(holidays)
        data_hourly.loc[:, "Weekday"] = data_hourly.index.day_of_week

        #Remove eccess data
        data_hourly = data_hourly[(data_hourly.index >= real_start) & (data_hourly.index < self.end)]
        return data_hourly


if __name__ == "__main__":
    start, end = "2018-05-01", "2018-11-01"
    df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=False)

    #Total_df = createTransformation(df, start, end).remove_outliers().getTotalData()
    #print(Total_df.to_string())

    Users = createTransformation(df, start, end)
    User_61 = Users.remove_outliers().getUserData(user="000000022")
    print(Users.data.userID)
