from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import timedelta

class ImportEV:
    def getCaltech(self, removeUsers = False, userSampleLimit = 50):
        data = pd.DataFrame(json.load(open(Path('../Data/acn_caltech.json'), 'r'))['_items'])

        if removeUsers:
            data = data.dropna(subset=['userID']).groupby(by="userID").filter(lambda x: len(x) > userSampleLimit)
        """
        for i in range(len(data["doneChargingTime"])):
            if data["doneChargingTime"][i] is None:
                data.loc[i, "doneChargingTime"] = data["disconnectTime"][i]

        data["connectionTime"] = pd.to_datetime(data["connectionTime"]) - timedelta(hours=7)
        data["disconnectTime"] = pd.to_datetime(data["disconnectTime"]) - timedelta(hours=7)
        data["doneChargingTime"] = pd.to_datetime(data["doneChargingTime"]) - timedelta(hours=7)
        """

        """
        Import Caltech
        :return:
        :rtype:
        """
        return data

    def getJPL(self):
        """
        Import JPL
        :return:
        :rtype:
        """

    def getOffice(self):
        """
        Import Office
        :return:
        :rtype:
        """

class ImportWeather:
    def getPasadena(self):
        """
        Import Pasadena Weather Data
        :return:
        :rtype:
        """

    def getSiliconValley(self):
        """
        Import Silicon Valley Weather Data
        :return:
        :rtype:
        """


if __name__ == "__main__":
    df = ImportEV().getCaltech(removeUsers=True, userSampleLimit=400)
    print(df["userID"])