import numpy as np
from scipy import interpolate


__all__ = ['create_grid', 'rotate_grid']

def create_grid(size, npix:float|int, type='linear'):
    if type.lower() == 'linear':
        x = np.linspace(-size, size, npix)
        X, Y, Z = np.meshgrid(x,x,x, indexing='ij')
    
    elif type.lower() == 'log':
        # create the positive half
        x_p = np.geomspace(1, size, npix//2)
        # negative half is just - the positive
        x_n = -1 * x_p
        # Then flip the negative half so it increases and append the positive half
        x = np.append(np.flip(x_n), x_p)
        X, Y, Z = np.meshgrid(x, x, x)

    elif type.lower() == 'spherical':
        i = input('This grid type does not work properly yet, are you sure you want to continue? [y/n]')
        if i == 'n' or i.lower() == 'no':
            raise KeyboardInterrupt ('interupted')
        # This spherical grid only returns nans for a reason I am not able to discover,
        theta = np.linspace(0, np.pi, npix)
        phi = np.linspace(0, 2*np.pi, npix)
        x = np.linspace(0, (size)**(1/2), npix) ** 2

        R, T, P = np.meshgrid(x, theta, phi)
        X = R * np.sin(T) * np.cos(P)
        Y = R * np.sin(T) * np.sin(P)
        Z = R * np.cos(T)
    else:
        raise ValueError ('Unknown grid type')
    return (X, Y, Z), x

def up_center_res(grid):
    npix = grid[0].shape[0]# The amount of pixels along one dimension of the grid, equal to the pixel_count variable in
    segments_outer = segment_3dgrid(grid)
  
    # extract the middle cube out of the segments. 
    middle_cube = segments_outer.pop(13)
    middle_cube_X = middle_cube[0]
    middle_cube_axis = np.linspace(np.min(middle_cube_X), np.max(middle_cube_X), int(npix))

    middle_cube_new = tuple(np.meshgrid(middle_cube_axis, middle_cube_axis, middle_cube_axis, indexing='ij'))
    segments = segments_outer + [middle_cube_new]
    return segments

def segment_3dgrid(grid:tuple):
    """This function will segment a 3d grid into 27 smaller grids. These grids are returned in an array where the first
    element stars at (0,0,0) corner increasing first along the Z-axis, then X-axis, then Y-axis

    Args:
        grid (tuple): The grid you want to segment
    """
    X, Y, Z = grid
    x = X[:,0,0]
    segments = []
    segment_size = (X.shape[0] // 3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                upper_lim_i = (i + 1) * segment_size + 1
                upper_lim_j = (j + 1) * segment_size + 1
                upper_lim_k = (k + 1) * segment_size + 1
                X_seg, Y_seg, Z_seg = np.meshgrid(x[i * segment_size : upper_lim_i], 
                                                x[j * segment_size : upper_lim_j],
                                                x[k * segment_size : upper_lim_k], indexing='ij')
                segments.append((X_seg, Y_seg, Z_seg))
    return segments

    
def rotate_grid(theta:float|int, phi:float|int, grid:tuple):
    """Rotates a 3d grid by theta degrees around the X-axis and then phi degrees around the Y axis

    Args:
        theta (float | int): _description_
        phi (float | int): _description_
        grid (tuple): _description_

    Returns:
        _type_: _description_
    """
    X, Y, Z = grid

    # First rotate around Z-axis (Theta)
    X_prime = X * np.cos(theta) - Y * np.sin(theta)
    Y_prime = X * np.sin(theta) + Y * np.cos(theta)
    Z_prime = Z.copy()

    # Then rotate around Y (phi)
    X_prime_prime = X_prime * np.cos(phi) + Z_prime * np.sin(phi)
    Y_prime_prime = Y_prime.copy()
    Z_prime_prime = X_prime * - np.sin(phi) + Z_prime * np.cos(phi)

    return X_prime_prime, Y_prime_prime, Z_prime_prime


def fibonacci_sphere(radius: float, num_points: int):
    ga = (3 - np.sqrt(5)) * np.pi # golden angle                                                                             

    # Create a list of golden angle increments along tha range of number of points                                           
    theta = ga * np.arange(num_points)

    # Z is a split into a range of -1 to 1 in order to create a unit circle                                                  
    z = np.linspace(1/num_points-1, 1-1/num_points, num_points)

    # a list of the radii at each height step of the unit circle                                                             
    r = np.sqrt(1 - z * z)

    # Determine where xy fall on the sphere, given the azimuthal and polar angles                                            
    y = radius * np.sin(theta) * r
    x = radius * np.cos(theta) * r

    return (x,y,z)