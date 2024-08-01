import swisseph as swe
import datetime
import os
import time
import numpy as np
import pandas as pd

ephemeris_path = "E:\Senior Project\ZPH" 
swe.set_ephe_path(ephemeris_path)


t = pd.read_csv('dataset\Spatial-Clustering_ctr_mag4_5upper.csv')
t['time'] = pd.to_datetime(t['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
t = t[['time']]

def get_julian_date(date_str):
    date_time = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    jd = swe.julday(date_time.year, date_time.month, date_time.day, 
                    date_time.hour + date_time.minute / 60 + date_time.second / 3600)
    return jd

t['jd'] = t['time'].apply(get_julian_date)

t.to_csv('juain_date405.csv', index=False)