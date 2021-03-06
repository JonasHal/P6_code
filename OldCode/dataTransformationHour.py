import pandas as pd
import numpy as np
from P6_code.FinishedCode.importData import ImportEV

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
        user_df["chargingTime"] = pd.to_datetime(user_df["doneChargingTime"]) - pd.to_datetime(
            user_df["connectionTime"])
        user_df = user_df[["connectionTime", "chargingTime", "kWhDelivered"]]

        # Feature Information on charging days
        date_index_charging = pd.to_datetime(user_df.pop("connectionTime"))

        user_df.loc[:, ("connectionHour")] = pd.to_datetime(date_index_charging.dt.strftime('%Y-%m-%d %H'))

        # Index creation
        periods = pd.to_datetime(self.end) - pd.to_datetime(self.start)
        index_values = pd.period_range(self.start, periods=periods.days*24, freq='H').to_timestamp()

        # Merge the days
        user_df = pd.merge(user_df, pd.DataFrame(data=pd.to_datetime(index_values), columns=["connectionHour"]), how="outer", on="connectionHour")
        user_df = user_df.sort_values(by="connectionHour")

        #Additional Feature Informations
        date_index = pd.to_datetime(user_df.pop("connectionHour"))

        user_df.loc[:, ("Year")] = date_index.dt.strftime('%Y')
        user_df.loc[:, ("Month")] = date_index.dt.strftime('%m')
        user_df.loc[:, ("Day")] = date_index.dt.strftime('%d')
        user_df.loc[:, ("Weekday")] = date_index.dt.day_of_week
        user_df.loc[:, ("Hour")] = date_index.dt.strftime('%H')

        #Impute 0 on missing values
        user_df["chargingTime"].fillna(pd.Timedelta('0 days'), inplace=True)
        user_df["kWhDelivered"].fillna(0, inplace=True)

        #Timedelta to numeric on ConnectionTime
        user_df['chargingTime'] = user_df['chargingTime'] / np.timedelta64(1, 's')

        return user_df.reset_index(drop=True)

if __name__ == "__main__":
    start, end = "2018-05-01", "2018-11-01"
    df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=True, userSampleLimit=30)
    Users = createUsers(df, start, end)
    User_61 = Users.getUserData(user="000000061")
    print(User_61.to_string())