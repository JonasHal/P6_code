import pandas as pd
from P6_code.FinishedCode.importData import ImportEV
from P6_code.FinishedCode.dataTransformation import createUsers
import plotly.graph_objects as go
import numpy as np

# 3-siwma-reglen kan bruges på både brugerniveau og for hele balladen.
# 0-observationer kan sorteres fra.


def three_sigma_outlier(observation):
    return abs(observation) > 3*std


def iqr_outlier(observation):
    return (observation < q1 - 1.5*iqr) or (observation > q3 + 1.5*iqr)


if __name__ == '__main__':
    start, end = '2000-01-01', '2022-12-31'
    df = ImportEV().getBoth(start_date=start, end_date=end, removeUsers=False)
    df["chargingTime"] = pd.to_datetime(df["doneChargingTime"]) - pd.to_datetime(df["connectionTime"])

    variable = "chargingTime"
    q1 = np.quantile(df[variable], 0.25)
    q3 = np.quantile(df[variable], 0.75)
    iqr = q3 - q1
    std = np.std(df[variable])

    # Global outliers:

    # for i in range(len(df)):
    #     if iqr_outlier(df.loc[i, 'chargingTime']):
    #         print(i, df.loc[i, 'chargingTime'])

    #print(df.loc[452, ])


    # for i in range(len(df)):
    #     if three_sigma_outlier(df['chargingTime'], df.loc[i, 'chargingTime']):
    #         print(i, df.loc[i, 'chargingTime'])


    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df.index, y=df[variable], mode='markers'))
    # fig.add_hline(y=200*10^12, line_dash="dot")
    # fig.show()

    # Local outliers:

    #print(df[df.doneChargingTime.isna()])

    #print(df.loc[7627:7630, ].to_string())

    #print(df.sort_values(by='chargingTime', ascending=False)['spaceID'].unique())

    #df = df[df['chargingTime'] > pd.Timedelta(0)]

    df['kWhPerHour'] = df['kWhDelivered']/(df['chargingTime'].dt.seconds*3600)

    print(df.sort_values(by='kWhPerHour').head(10).to_string())
    print(df.sort_values(by='kWhPerHour').tail(10).to_string())
    #print(df['kWhDelivered']/df['chargingTime'].dt.seconds)

    #print(users.head().to_string())
