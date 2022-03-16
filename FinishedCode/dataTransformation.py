import pandas as pd
from FinishedCode.importData import ImportEV

# Preprocessing Functions
class createUser:
    def __init__(self, dataframe):
        self.data = dataframe

    def transformData(self, start_date, end_date, user):
        #Slice the data to one user:
        user_df = self.data[self.data["userID"] == user]
        user_df = user_df[["connectionTime", "doneChargingTime", "kWhDelivered"]] #TODO: Når preproces er lavet så ændre til chargingTime
        date_index = pd.to_datetime(user_df.connectionTime)
        user_df["Hour"] = date_index.dt.strftime('%H')
        user_df["Minute"] = date_index.dt.strftime('%M')
        user_df["Second"] = date_index.dt.strftime('%S')

        user_df = user_df.set_index("connectionTime")
        print(user_df)

        #Index creation
        periods = pd.to_datetime(end_date) - pd.to_datetime(start_date)
        index_values = (pd.date_range(start_date, periods=periods.days, freq='D'))
        #print(index_values)

        return

if __name__ == "__main__":
    start, end = "2018-05-01", "2018-11-01"
    df = ImportEV().getCaltech(start_date=start, end_date=end, removeUsers=True, userSampleLimit=30)
    User_1 = createUser(df).transformData(start_date=start, end_date=end, user="000000061")
    #print(df)