import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np
from P6_code.FinishedCode.importData import ImportEV

#Data Transformation for each day
pd.options.mode.chained_assignment = None
# Preprocessing Functions
class createUsers:
    def __init__(self, dataframe, start, end):
        self.data = dataframe
        self.start = start
        self.end = end

    def getUserData(self, user):
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

    def remove_outliers(self):
        q1, q3 = np.quantile(self.data['chargingTime'], [0.25, 0.75])
        self.data = self.data[self.data['chargingTime'] < q3 + 1.5 * (q3 - q1)]
        return self

if __name__ == "__main__":
    start, end = "2018-05-01", "2018-11-01"
    df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=True, userSampleLimit=30)
    Users = createUsers(df, start, end)
    User_61 = Users.remove_outliers().getUserData(user="000000022")
    print(User_61.to_string())