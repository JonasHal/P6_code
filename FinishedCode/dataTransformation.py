import pandas as pd
import numpy as np
from P6_code.FinishedCode.importData import ImportEV

# Preprocessing Functions
class createUser:
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

        user_df.loc[:, ("connectionDay")] = pd.to_datetime(date_index_charging.dt.strftime('%Y-%m-%d'))

        # Index creation
        periods = pd.to_datetime(self.end) - pd.to_datetime(self.start)
        index_values = (pd.date_range(self.start, periods=periods.days, freq='D'))

        # Merge the days
        user_df = pd.merge(user_df, pd.DataFrame(data=pd.to_datetime(index_values), columns=["connectionDay"]), how="outer", on="connectionDay")
        user_df = user_df.sort_values(by="connectionDay")

        #Additional Feature Informations
        date_index = pd.to_datetime(user_df.pop("connectionDay"))

        user_df.loc[:, ("Year")] = date_index.dt.strftime('%Y')
        user_df.loc[:, ("Month")] = date_index.dt.strftime('%m')
        user_df.loc[:, ("Day")] = date_index.dt.strftime('%d')
        user_df.loc[:, ("Weekday")] = date_index.dt.strftime('%A')

        #Impute 0 on missing values
        user_df["chargingTime"].fillna(pd.Timedelta('0 days'), inplace=True)
        user_df["kWhDelivered"].fillna(0, inplace=True)

        return user_df.reset_index(drop=True)

if __name__ == "__main__":
    start, end = "2018-05-01", "2018-11-01"
    df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=True, userSampleLimit=30)
    Users = createUser(df, start, end)
    User_61 = Users.getUserData(user="000000061")
    print(User_61.to_string())