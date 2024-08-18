import swisseph as swe
import datetime
import os
import time
import numpy as np
import pandas as pd

ephemeris_path = "E:\Senior Project\ZPH" 

t = pd.read_csv('DataSet\edata.csv')
t = t #.iloc[:500, :]
swe.set_ephe_path(ephemeris_path)

def get_asteroid_position(date_str, asteroid_number):
    date_time = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    jd = swe.julday(date_time.year, date_time.month, date_time.day, 
                    date_time.hour + date_time.minute / 60 + date_time.second / 3600)

    asteroid_id = swe.AST_OFFSET + asteroid_number
    position, _ = swe.calc_ut(jd, asteroid_id, swe.FLG_XYZ)
    
    return [
        position[0],
        position[1],
        position[2]
    ]

asteroid_number = 1  # For Ceres
start = time.time()

t[[f'{asteroid_number}_x', f'{asteroid_number}_y', f'{asteroid_number}_z']] = t['time'].apply(
    lambda date_str: pd.Series(get_asteroid_position(date_str, asteroid_number))
)
t.to_csv('asteroid78.csv', index=False)
end = time.time()
print(t)

print('Time: ', end-start)