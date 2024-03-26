from multiprocessing import Value
import os
from pyexpat import model
from starwinds_readplt.dataset import Dataset
import numpy as np
data_loc = os.environ['FMPdata']


def find_file(name, path):
    """This function makes use of glob.glob to find the file(s) named file, in directory path

    Args:
        name (str): _description_
        path (str): _description_

    Returns:
        str: The location of the first file
    """
    import glob
    files = glob.glob(data_loc + '/' + path + '/**/' + name, recursive=True)
    print('{}'.format(f'Reading from file: {files[0]}'))
    return files[0]

def read_model(model_name:str):
    try:
        model_path = find_file('3d__var_3_n00060000.plt', model_name)
    except IndexError:
        print('3d__var_3_n00060000.plt not found, trying 3d__var_3_n00006000.plt')
        model_path = find_file('3d__var_3_n00006000.plt', model_name)

    ds = Dataset.from_file(model_path)
    try:
        params_file = open(os.path.join(data_loc, model_name + '/SCPARAM.in'), 'r')
        model_params = params_file.read()
        params_file.close()
    except:
        print("No SCPARAM.in found.")
        model_params = None
    try:
        star_file = open(os.path.join(data_loc, model_name + '/STAR.in'), 'r')
        param_lines = star_file.readlines()
        star_params = {}
        for n, line in enumerate(param_lines[-3:]):
            split = line.split('  ')
            value = split[0]
            name = split[-1].split(' ')
            if name[0].isalpha() == False:
                name = split[-1].split(' ')[1]
            else:
                name = name[0]
            star_params[str(name)] = float(value)
    except:
        print("No STAR.in found. Defaulting to solar parameters")
        star_params = {'RadiusStar': 1, 'MassStar': 1, "RotationPeriodStar": 24.47}
    return ds, model_params, star_params


def import_data(name:str, interpolate='nearest', full_output=False):
    assert type(full_output) == bool
    """This function imports the data of the star given in the name parameter. It uses the files from the model to get the grid into units of cm such that it can be used with CGS


    Args:
        name (str): Name of the model 
        interpolate (str, optional): _description_. Defaults to 'nearest'.
        full_output (bool, optional): If True, also returns the data points used to create the interpolater, and returns the model_params

    Returns:
        _type_: _description_
    """
    import astropy.units as u
    from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
    from scipy.spatial.qhull import QhullError
    ds, model_params, star_params = read_model(name)
    variable_list = ds.variables
    # Get a stack of points and the data in a numpy format and create an interpolator function
    ds_points = np.stack([ds(name) for name in ds.variables[:3]], axis=-1)
    radius_star = star_params['RadiusStar'] * u.R_sun
    star_params['RadiusStar'] = radius_star.to(u.cm).value
    ds_points *= radius_star.to(u.cm).value
    ds_data = np.stack([ds(name) for name in ds.variables], axis=-1)

    if not full_output:
        if interpolate.lower() == 'nearest':
            return NearestNDInterpolator(ds_points, ds_data), variable_list, star_params, None, None, None
        elif interpolate.lower() == 'linear':
            try:
                return LinearNDInterpolator(ds_points, ds_data), variable_list, star_params, None, None, None
            except QhullError:
                print("QhullError: Linear interpolation failed, \
                Using nearest interpolation") 
                return NearestNDInterpolator(ds_points, ds_data), variable_list, star_params, None, None, None
        else:
            raise ValueError
    else:
        if interpolate.lower() == 'nearest':
            return NearestNDInterpolator(ds_points, ds_data),  variable_list, star_params, model_params, ds_points, ds_data
        elif interpolate.lower() == 'linear':
            try:
                return LinearNDInterpolator(ds_points, ds_data), variable_list, star_params, model_params, ds_points, ds_data
            except QhullError:
                print("QhullError: Linear interpolation failed, \
                Using nearest interpolation") 
                return NearestNDInterpolator(ds_points, ds_data),  variable_list, star_params, model_params, ds_points, ds_data
        else:
            raise ValueError
    