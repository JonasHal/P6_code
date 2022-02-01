from pathlib import Path
from pandas import read_csv

data_1hour = read_csv(Path("../Data/electricitybalancenonv_2_1_2022.csv"), sep=',', decimal='.', encoding='latin1')
data_5mil = read_csv(Path("../Data/electricityprodex5minrealtime_2_1_2022.csv"), sep=',', decimal='.', encoding='latin1')
data_consump = read_csv(Path("../Data/consumptiondk3619codehour_2_1_2022.csv"), sep=',', decimal='.', encoding='latin1')

print(data_1hour.head().to_string())
print(data_5mil.head().to_string())
print(data_consump.head().to_string())