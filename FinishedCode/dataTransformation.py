import pandas as pd
from FinishedCode.importData import ImportEV

# Preprocessing Functions
class createUser:
    def __init__(self, dataframe):
        self.data = dataframe

    def transformData(self, user):


if __name__ == "__main__":
    df = ImportEV().getCaltech(removeUsers=True, userSampleLimit=50)
    User_1 = createUser(df).transformData()