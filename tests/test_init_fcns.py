import matplotlib
matplotlib.use('Agg')
from modules import *
import pandas as pd

def test_polar_to_xy():
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
    
    result_1a = polar_to_xy(pos_info_a, pa, asec = False, south = True)
    assert result_1a["x_pix_coord"][0] == 0
    assert result_1a["y_pix_coord"][0] == 10

    result_1b = polar_to_xy(pos_info_b, pa, asec = False, south = True)
    assert result_1b["x_pix_coord"][0] == -10
    assert result_1b["y_pix_coord"][0] == 0

    # tests with PA = 40
    pa = 40
    
    result_2a = polar_to_xy(pos_info_a, pa, asec = False, south = True)
    assert result_2a["x_pix_coord"][0] == 0
    assert result_2a["y_pix_coord"][0] == 10

    result_2b = polar_to_xy(pos_info_b, pa, asec = False, south = True)
    assert result_2b["x_pix_coord"][0] == -10
    assert result_2b["y_pix_coord"][0] == 0


    # tests with PA = -30
    pa = -30
    
    result_3a = polar_to_xy(pos_info_a, pa, asec = False, south = True)
    assert result_3a["x_pix_coord"][0] == 0
    assert result_3a["y_pix_coord"][0] == 10

    result_3b = polar_to_xy(pos_info_b, pa, asec = False, south = True)
    assert result_3b["x_pix_coord"][0] == -10
    assert result_3b["y_pix_coord"][0] == 0

    
    # a test in the North    
    result_3a_n = polar_to_xy(pos_info_a, pa, asec = False, north = True)
    assert result_3a_n["x_pix_coord"][0] == 0
    assert result_3a_n["y_pix_coord"][0] == 10

    result_3b_n = polar_to_xy(pos_info_b, pa, asec = False, north = True)
    assert result_3b_n["x_pix_coord"][0] == -10
    assert result_3b_n["y_pix_coord"][0] == 0
    

    return



'''
def test_config():
    assert edification.graft_feh() == '22'

def test_getcwd():
    #junktest_compile_normalizations.compile_bkgrnd()
    # configuration data
    graft_phases.yada_yada() == '22'
    assert True

# check if the directory-making function works
def test_make_dirs():

    # call function to make directories
    make_dirs()

    # do all the directories exist now?
    for vals in config["data_dirs"]:
        abs_path_name = str(config["data_dirs"][vals])
        assert os.path.exists(abs_path_name)

# test if the phase region boundaries are being read in correctly
def test_phase_regions():

    min_good_phase, max_good_phase = phase_regions()

    # is min smaller than max
    assert min_good_phase < max_good_phase

    # are the phases interpreted as floats
    assert isinstance(min_good_phase,float)
'''
