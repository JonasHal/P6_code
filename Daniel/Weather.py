import pandas as pd
from pathlib import Path

weather_caltech = pd.read_csv(Path('../Data/weather_Pasadena_hourly.csv'))
weather_caltech.drop([col for col in weather_caltech.columns if 'qc' in col], axis='columns', inplace=True)
weather_caltech.drop(['Stn Id', 'Stn Name', 'CIMIS Region'], axis='columns', inplace=True)
weather_caltech['Date'] = pd.to_datetime(weather_caltech['Date'])
agg_dict = {col: 'mean' for col in weather_caltech.columns.drop('Date')}
agg_dict['Precip (mm)'] = 'sum'

weather_caltech = weather_caltech.groupby('Date').agg(agg_dict)
weather_caltech.drop('Hour (PST)', axis='columns', inplace=True)

#print(weather_caltech.head(20).to_string())

weather_silicon = pd.read_csv(Path('../Data/weather_Pasadena_hourly.csv'))
weather_silicon.drop([col for col in weather_silicon.columns if 'qc' in col], axis='columns', inplace=True)
weather_silicon.drop(['Stn Id', 'Stn Name', 'CIMIS Region'], axis='columns', inplace=True)
weather_silicon['Date'] = pd.to_datetime(weather_silicon['Date'])
agg_dict = {col: 'mean' for col in weather_silicon.columns.drop('Date')}
agg_dict['Precip (mm)'] = 'sum'

weather_silicon = weather_silicon.groupby('Date').agg(agg_dict)
weather_silicon.drop('Hour (PST)', axis='columns', inplace=True)

print(weather_silicon.head(20).to_string())