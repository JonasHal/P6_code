import pandas as pd

# Preprocessing Functions
def imputeNA(data, i, column='TotalLoad'):
    other_observations = 0
    for t in range(7):
        if t != 3:
            other_observations += data[column][data.index[i] - pd.Timedelta(t - 3, 'd')]
    data.loc[data.index[i], column] = round(other_observations / 6, 2)
    return data

def fillnaTotalLoad(data):
    for i in range(len(data)):
        if pd.isna(data.TotalLoad[data.index[i]]):
            data = imputeNA(data, i)
    return data

def changeOutliers(data, columns):
    for i in range(len(data)):
        if data.TotalLoad[data.index[i]] > 10000:
            for column in columns:
                data = imputeNA(data, i, column)
    return data

def changeNA(data, columns):
    for i in range(len(data)):
        if pd.isna(data.ExchangeNordicCountries[data.index[i]]):
            for column in columns:
                if pd.isna(data[column][data.index[i]]):
                    data = imputeNA(data, i, column)
    return data