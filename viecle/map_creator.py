import numpy as np

def map_creator(i=1):
    pi = np.pi * 2
    if i == 1:
        return  np.array([
            [35., 15.7, 31.4, 15.7, 35., 62.8, 22.6, 11.8, 23.6, 23.6, 23.6, 11.8, 22.6, 62.8],
            [0.,  -30., 30., -30.,  0., -20.,  0.,  -15.,  15., -15.,  15., -15.,  0.,  -20.]
            ])

    if i == 2:
        r = 25
        q1 = 2*r*pi*0.25
        q3 = 2*r*pi*0.75
        return  np.array([
            [r, q1, q3, q1, r*2, q1, q3, q1, r*2, q1, q3, q1, r*2, q1, q3, q1, r],
            [0., r, -r,  r,  0.,  r, -r,  r,  0.,  r, -r,  r,  0,  r,  -r,  r, 0]
            ])



