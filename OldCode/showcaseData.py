from pathlib import Path
from pandas import read_csv

data_5min = read_csv(Path("../Data/electricityprodex5minrealtime_2_1_2022.csv"), sep=',', decimal='.', encoding='latin1')
data_1hour = read_csv(Path("../Data/electricitybalancenonv_2_1_2022.csv"), sep=',', decimal='.', encoding='latin1')
data_consump = read_csv(Path("../Data/consumptiondk3619codehour_2_1_2022.csv"), sep=',', decimal='.', encoding='utf')

data = [data_5min, data_1hour, data_consump]

for i in data:
    print(i.head().to_string())
    print(i.describe().to_string())