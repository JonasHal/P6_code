from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import timedelta

class ImportEV:
    def getCaltech(self, start_date, end_date, removeUsers = False, userSampleLimit = 50):
        """
        Import Caltech
        :return:
        :rtype:
        """
        data = pd.DataFrame(json.load(open(Path('../Data/acn_caltech.json'), 'r'))['_items'])
        self.dataName = "Caltech"

        for i in range(len(data["doneChargingTime"])):
            if data["doneChargingTime"][i] is None:
                data.loc[i, "doneChargingTime"] = data["disconnectTime"][i]

        data["connectionTime"] = pd.to_datetime(data["connectionTime"]) - timedelta(hours=7)
        data["disconnectTime"] = pd.to_datetime(data["disconnectTime"]) - timedelta(hours=7)
        data["doneChargingTime"] = pd.to_datetime(data["doneChargingTime"]) - timedelta(hours=7)
        data = data[(data.connectionTime > start_date) & (data.connectionTime < end_date)]

        data["chargingTime"] = pd.to_datetime(data["doneChargingTime"]) - pd.to_datetime(data["connectionTime"])

        if removeUsers:
            data = data.dropna(subset=['userID']).groupby(by="userID").filter(lambda x: len(x) > userSampleLimit)

        return data.reset_index(drop=True)

    def getJPL(self, start_date, end_date, removeUsers = False, userSampleLimit = 50):
        """
        Import JPL
        :return:
        :rtype:
        """
        data = pd.DataFrame(json.load(open(Path('../Data/acn_jpl.json'), 'r'))['_items'])
        self.dataName = "JPL"

        for i in range(len(data["doneChargingTime"])):
            if data["doneChargingTime"][i] is None:
                data.loc[i, "doneChargingTime"] = data["disconnectTime"][i]

        data["connectionTime"] = pd.to_datetime(data["connectionTime"]) - timedelta(hours=7)
        data["disconnectTime"] = pd.to_datetime(data["disconnectTime"]) - timedelta(hours=7)
        data["doneChargingTime"] = pd.to_datetime(data["doneChargingTime"]) - timedelta(hours=7)
        data = data[(data.connectionTime > start_date) & (data.connectionTime < end_date)]

        data["chargingTime"] = pd.to_datetime(data["doneChargingTime"]) - pd.to_datetime(data["connectionTime"])

        if removeUsers:
            data = data.dropna(subset=['userID']).groupby(by="userID").filter(lambda x: len(x) > userSampleLimit)

        return data.reset_index(drop=True)

    def getOffice(self, start_date, end_date, removeUsers = False, userSampleLimit = 50):
        """
        Import Office
        :return:
        :rtype:
        """
        data = pd.DataFrame(json.load(open(Path('../Data/acn_office1.json'), 'r'))['_items'])
        self.dataName = "Office"

        for i in range(len(data["doneChargingTime"])):
            if data["doneChargingTime"][i] is None:
                data.loc[i, "doneChargingTime"] = data["disconnectTime"][i]

        data["connectionTime"] = pd.to_datetime(data["connectionTime"]) - timedelta(hours=7)
        data["disconnectTime"] = pd.to_datetime(data["disconnectTime"]) - timedelta(hours=7)
        data["doneChargingTime"] = pd.to_datetime(data["doneChargingTime"]) - timedelta(hours=7)
        data = data[(data.connectionTime > start_date) & (data.connectionTime < end_date)]

        data["chargingTime"] = pd.to_datetime(data["doneChargingTime"]) - pd.to_datetime(data["connectionTime"])

        if removeUsers:
            data = data.dropna(subset=['userID']).groupby(by="userID").filter(lambda x: len(x) > userSampleLimit)

        return data.reset_index(drop=True)

    def getBoth(self, start_date, end_date, removeUsers = False, userSampleLimit = 50):
        """
        Import both Caltech and JPL and concatenate
        :return:
        :rtype:
        """
        caltech = pd.DataFrame(json.load(open(Path('../Data/acn_caltech.json'), 'r'))['_items'])
        jpl = pd.DataFrame(json.load(open(Path('../Data/acn_jpl.json'), 'r'))['_items'])

        data = pd.concat([caltech, jpl], ignore_index=True)

        for i in range(len(data["doneChargingTime"])):
            if data["doneChargingTime"][i] is None:
                data.loc[i, "doneChargingTime"] = data["disconnectTime"][i]

        data["connectionTime"] = pd.to_datetime(data["connectionTime"]) - timedelta(hours=7)
        data["disconnectTime"] = pd.to_datetime(data["disconnectTime"]) - timedelta(hours=7)
        data["doneChargingTime"] = pd.to_datetime(data["doneChargingTime"]) - timedelta(hours=7)
        data = data[(data.connectionTime > start_date) & (data.connectionTime < end_date)]

        data["chargingTime"] = pd.to_datetime(data["doneChargingTime"]) - pd.to_datetime(data["connectionTime"])

        data = data.sort_values(by="connectionTime").reset_index(drop=True)

        if removeUsers:
            data = data.dropna(subset=['userID']).groupby(by="userID").filter(lambda x: len(x) > userSampleLimit)

        return data.reset_index(drop=True)


class ImportWeather:
    def getPasadena(self, agg='hour'):
        """
        Import Pasadena Weather Data
        :return:
        :rtype:
        """
        weather_pasadena = pd.read_csv(Path('../Data/weather_Pasadena_hourly.csv'))
        weather_pasadena.drop([col for col in weather_pasadena.columns if 'qc' in col], axis='columns', inplace=True)
        weather_pasadena.drop(['Stn Id', 'Stn Name', 'CIMIS Region'], axis='columns', inplace=True)
        weather_pasadena['Date'] = pd.to_datetime(weather_pasadena['Date'])

        if agg == 'day':
            agg_dict = {col: 'mean' for col in weather_pasadena.columns.drop('Date')}
            agg_dict['Precip (mm)'] = 'sum'
    
            weather_pasadena = weather_pasadena.groupby('Date').agg(agg_dict)
            weather_pasadena.drop('Hour (PST)', axis='columns', inplace=True)

        return weather_pasadena

    def getSiliconValley(self, agg='hour'):
        """
        Import Silicon Valley Weather Data
        :return:
        :rtype:
        """
        weather_silicon = pd.read_csv(Path('../Data/weather_Pasadena_hourly.csv'))
        weather_silicon.drop([col for col in weather_silicon.columns if 'qc' in col], axis='columns', inplace=True)
        weather_silicon.drop(['Stn Id', 'Stn Name', 'CIMIS Region'], axis='columns', inplace=True)
        weather_silicon['Date'] = pd.to_datetime(weather_silicon['Date'])

        if agg == 'day':
            agg_dict = {col: 'mean' for col in weather_silicon.columns.drop('Date')}
            agg_dict['Precip (mm)'] = 'sum'

            weather_silicon = weather_silicon.groupby('Date').agg(agg_dict)
            weather_silicon.drop('Hour (PST)', axis='columns', inplace=True)

        return weather_silicon


if __name__ == "__main__":
    df = ImportEV().getBoth(start_date="2001-10-01", end_date="2022-10-04", removeUsers=True)
    #df = ImportEV().getJPL(start_date="2018-05-01", end_date="2018-11-01", removeUsers=True, userSampleLimit=10)
    #df = ImportEV().getBoth(start_date="2018-05-01", end_date="2018-11-01", removeUsers=True, userSampleLimit=1)
    print(df.columns)