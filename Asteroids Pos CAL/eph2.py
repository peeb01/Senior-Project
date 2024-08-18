import swisseph as swe
import numpy as np
import pandas as pd
import time

ephemeris_path = r"E:\Senior Project\ZPH"


swe.set_ephe_path(ephemeris_path)
t = pd.read_csv('Asteroids Pos\juain_date405.csv')

start = time.time()
def get_asteroid_position(jds, asteroid_number):
    asteroid_id = swe.AST_OFFSET + asteroid_number
    positions = []
    for jd in jds:
        position, _ = swe.calc_ut(jd, asteroid_id, swe.FLG_XYZ | swe.FLG_HELCTR)    # Ref from Earth
        positions.append(position)
    positions = np.array(positions)
    return positions

jds = t['jd'].values
asteroid_number = 1
positions = get_asteroid_position(jds, asteroid_number)

t[f'{asteroid_number}_x'] = positions[:, 0]
t[f'{asteroid_number}_y'] = positions[:, 1]
t[f'{asteroid_number}_z'] = positions[:, 2]

print('Times : ' ,time.time() - start)
print(t)