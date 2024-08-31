import swisseph as swe
import numpy as np
import pandas as pd
import time

ephemeris_path = r"E:\Senior Project\ZPH"

swe.set_ephe_path(ephemeris_path)

t = pd.read_csv('Asteroids Pos CAL\juain_date405.csv')

start = time.time()

def get_planet_position(jds, planet):
    positions = []
    for jd in jds:

        position, _ = swe.calc_ut(jd, planet, swe.FLG_XYZ)
        positions.append(position)
    positions = np.array(positions)
    return positions


jds = t['jd'].values

planets = [swe.SUN, swe.MOON, swe.MERCURY, swe.VENUS, swe.MARS, swe.JUPITER, swe.SATURN, swe.URANUS, swe.NEPTUNE]

for planet in planets:
    positions = get_planet_position(jds, planet)

    t[f'{swe.get_planet_name(planet)}_x'] = positions[:, 0]
    t[f'{swe.get_planet_name(planet)}_y'] = positions[:, 1]
    t[f'{swe.get_planet_name(planet)}_z'] = positions[:, 2]

print('Times :', time.time() - start)
print(t)

t.to_csv('position_plannet.csv', index=False)