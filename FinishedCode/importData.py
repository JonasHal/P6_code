from pathlib import Path
import pandas as pd
import numpy as np

import FinishedCode.preprocessingfunctions as pf

class ImportEV:
    def getCaltech(self):
        """
        Import Caltech
        :return:
        :rtype:
        """

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

    def getNewlyReg(self):
        """
        Import Newly Registered Cars
        :return:
        :rtype:
        """

    def getTotalReg(self):
        """
        Import Total Registered Cars
        :return:
        :rtype:
        """

class ImportEnerginet:

    #Getters
    def getElectricityBalanced(self):
        """
        The Dataset with the Energy balance in Denmark
        :return:
        :rtype:
        """
        #Dette er det eneste vi bruger lige nu, men regner med at få det hele igang på et tidspunkt

        #Skulle tilgå Databasen og hente data derfra: Nu er det bare fra CSV
        data_1hour = pd.read_csv(Path("../Data/electricitybalancenonv_2_1_2022.csv"), sep=',', decimal='.',
                                 encoding='utf-8', index_col='HourUTC', parse_dates=True, dtype={"PriceArea" : "string"})
        data_1hour = data_1hour.drop('HourDK', axis=1)

        # Splits the data into the two price areas DK1 and DK2
        data_dk1 = data_1hour[data_1hour['PriceArea'] == 'DK1']
        data_dk1 = data_dk1.sort_index().asfreq('1H')

        data_dk2 = data_1hour[data_1hour['PriceArea'] == 'DK2'] #TODO: Implementer så DK2 også får en prediction
        data_dk2 = data_dk2.sort_index().asfreq('1H')

        #Data Preprocessing
        data_dk1 = pf.fillnaTotalLoad(data_dk1)
        data_dk1 = data_dk1.drop('PriceArea', axis=1)
        data_dk1 = pf.changeOutliers(data_dk1, columns=['TotalLoad', 'Biomass', 'FossilGas', 'FossilHardCoal', 'FossilOil',
                                                     'OtherRenewable', 'Waste'])
        data_dk1 = pf.changeNA(data_dk1, data_dk1.columns)

        #Transform the Timedate index into features:
        date_index = data_dk1.index
        Danish_Holidays = ["2020-01-01", "2020-04-09", "2020-04-10", "2020-04-12", "2020-04-13", "2020-05-08", "2020-05-21", "2020-05-31", "2020-06-01", "2020-06-05", "2020-12-24", "2020-12-25", "2020-12-26", "2021-01-01", "2021-04-01", "2021-04-02", "2021-04-04", "2021-04-05", "2021-04-30", "2021-05-13", "2021-05-23", "2021-05-24", "2021-06-05", "2021-12-24", "2021-12-25", "2021-12-26"]
        data_dk1["Year"] = date_index.strftime('%Y')
        data_dk1["Month"] = date_index.strftime('%m')
        data_dk1["Day"] = date_index.strftime('%d')
        data_dk1["Hour"] = date_index.strftime('%H')
        data_dk1["Weekday"] = date_index.day_of_week
        data_dk1["Holiday"] = np.isin(date_index.strftime('%Y-%m-%d'), Danish_Holidays)

        return data_dk1[["TotalLoad", "Year", "Month", "Day", "Weekday", "Holiday", "Hour"]]


    def getIndustryConsumption(self):
        """
        The Dataset with the Consumption in each Industry
        :return:
        :rtype:
        """

    def getMunicipalityConsumption_month(self):
        """
        The Dataset with the Consumption in each Municipality and categories
        :return:
        :rtype:
        """

    def getMunicipalityConsumption_hour(self):
        """
        The Dataset with the Consumption in each Municipality and categories
        :return:
        :rtype:
        """

    def getMunicipiliyConsumption_Prediction(self):
        """
        The Dataset with the Consumption Prediction for 2025 in each Industry
        :return:
        :rtype:
        """

    def getMunicipility_Capacity(self):
        """
        The Dataset with the Capacity in each Municipility each month
        :return:
        :rtype:
        """

    #Orker ikke at sætte resten ind for nu :)

if __name__ == "__main__":
    df = ImportEnerginet().getElectricityBalanced()
    print(df)