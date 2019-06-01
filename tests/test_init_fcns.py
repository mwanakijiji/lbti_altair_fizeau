import matplotlib
matplotlib.use('Agg')
from modules import *
import pandas as pd

def test_polar_to_xy_southern():
    '''
    polar_to_xy() is supposed to

    Convert polar vectors (deg, pix) to xy vectors (pix, pix)
    which incorporate the parallactic angle
    (Note degrees are CCW from +y axis)

    INPUTS:
    pos_info: dictionary with keys
        "rad_pix": radius in pixels (if asec = False)
        "rad_asec": radius in arcsec (if asec = True)
        "angle_deg_EofN": angle in degrees E of true N
    pa: parallactic angle (or if no rotation compensation
        desired, just use 0)
    south: flag as to whether target is in south
    north: flag as to whether target is in south

    OUTPUTS:
    dictionary with the addition of keys
        "x_pix_coord": position in x in pixels (measured from host star at center)
        "y_pix_coord": position in y in pixels (measured from host star at center)
    '''

    # fake companion parameters
    d_a = {"rad_pix": [10], "angle_deg_EofN": [0]}
    d_b = {"rad_pix": [10], "angle_deg_EofN": [90]}
    pos_info_a = pd.DataFrame(data=d_a)
    pos_info_b = pd.DataFrame(data=d_b)
    
    # tests with PA = 0
    pa = 0
    
    result_1a = polar_to_xy(pos_info_a, pa, asec = False, south = True) # E of N: zero
    assert round(result_1a["x_pix_coord"][0],2) == round(0,2)
    assert round(result_1a["y_pix_coord"][0],2) == round(10,2)

    result_1b = polar_to_xy(pos_info_b, pa, asec = False, south = True) # E of N: +90
    assert round(result_1b["x_pix_coord"][0],2) == round(-10,2)
    assert round(result_1b["y_pix_coord"][0],2) == round(0,2)

    # tests with PA = -40
    pa = -40
    
    result_2a = polar_to_xy(pos_info_a, pa, asec = False, south = True) # E of N: zero
    assert round(result_2a["x_pix_coord"][0],2) == 6.43
    assert round(result_2a["y_pix_coord"][0],2) == 7.66

    result_2b = polar_to_xy(pos_info_b, pa, asec = False, south = True) # E of N: +90
    assert round(result_2b["x_pix_coord"][0],2) == -7.66
    assert round(result_2b["y_pix_coord"][0],2) == 6.43


    # tests with PA = +30
    pa = 30
    
    result_3a = polar_to_xy(pos_info_a, pa, asec = False, south = True) # E of N: zero
    assert round(result_3a["x_pix_coord"][0],2) == round(-5,2)
    assert round(result_3a["y_pix_coord"][0],2) == 8.66

    result_3b = polar_to_xy(pos_info_b, pa, asec = False, south = True) # E of N: +90
    assert round(result_3b["x_pix_coord"][0],2) == -8.66
    assert round(result_3b["y_pix_coord"][0],2) == round(-5,2)

    
    # a test in the North, PA = +150
    pa = 150
    
    result_3a_n = polar_to_xy(pos_info_a, pa, asec = False, south = False, north = True) # E of N: zero
    assert round(result_3a_n["x_pix_coord"][0],2) == round(5,2)
    assert round(result_3a_n["y_pix_coord"][0],2) == -8.66

    result_3b_n = polar_to_xy(pos_info_b, pa, asec = False, north = True) # E of N: +90
    assert round(result_3b_n["x_pix_coord"][0],2) == 8.66
    assert round(result_3b_n["y_pix_coord"][0],2) == round(5,2)

    return
