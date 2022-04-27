import pandas as pd
from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createUsers
import plotly.graph_objects as go
import numpy as np

# 3-siwma-reglen kan bruges på både brugerniveau og for hele balladen.
# 0-observationer kan sorteres fra.


def three_sigma_outlier(data, observation):
    return abs(observation) > 3*np.std(data)


def iqr_outlier(data, observation):
    iqr = np.quantile(data, 0.75) - np.quantile(data, 0.25)
    return (observation < np.quantile(data, 0.25) - 1.5*iqr) or (observation > np.quantile(data, 0.75) + 1.5*iqr)


if __name__ == '__main__':
    start, end = '2000-01-01', '2022-12-31'
    df = ImportEV().getBoth(start_date=start, end_date=end, removeUsers=False)
    df["chargingTime"] = pd.to_datetime(df["doneChargingTime"]) - pd.to_datetime(df["connectionTime"])

    # Global outliers:

    # for i in range(len(df)):
    #     if iqr_outlier(df['chargingTime'], df.loc[i, 'chargingTime']):
    #         print(i, df.loc[i, 'chargingTime'])

    #print(df.loc[452, ])


    # for i in range(len(df)):
    #     if three_sigma_outlier(df['chargingTime'], df.loc[i, 'chargingTime']):
    #         print(i, df.loc[i, 'chargingTime'])



    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["chargingTime"], mode='markers'))
    fig.show()

    # Local outliers:

    #print(users.head().to_string())
