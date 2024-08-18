import swisseph as swe
import numpy as np
import pandas as pd
import time
import warnings

warnings.filterwarnings('ignore')

ephemeris_path = r"E:\Senior Project\ZPH"
swe.set_ephe_path(ephemeris_path)

asteroids = pd.read_csv('Asteroids Pos CAL\\NEA_notna.csv')
asteroids = asteroids.sort_values(by='H')

as_num = asteroids['Number'].astype(int).values[:2500]


t = pd.read_csv('Asteroids Pos CAL\juain_date405.csv')
ts = t.copy()

def get_asteroid_position(jds, asteroid_number):
    asteroid_id = swe.AST_OFFSET + asteroid_number
    positions = []
    for jd in jds:
        try:
            position, _ = swe.calc_ut(jd, asteroid_id, swe.FLG_XYZ)
            positions.append(position)
        except Exception as e:
            # print(f"Error calculating position for JD {jd} and asteroid {asteroid_number}: {e}")
            continue
    positions = np.array(positions)
    return positions

jds = t['jd'].values

start = time.time()

n = 0
s = 0

error = []
for asteroid_number in as_num:
    try:
        positions = get_asteroid_position(jds, asteroid_number)
        if positions.size == 0:
            continue
        t[f'{asteroid_number}'] = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2)
        
        print(f'\033[92m{n}. Complete {asteroid_number}.\033[0m')
    except Exception as e:
        print(f"\033[91mError to Calculate position of {asteroid_number}...Next Asteroids.\033[0m")
        error.append(asteroid_number)
        n += 1
        continue

    n += 1

end = time.time()

t.to_csv(f'asteroid_distance_from_{n}.csv', index=False)

print(t)
print('Time: ', end - start, '\n')
print(error)