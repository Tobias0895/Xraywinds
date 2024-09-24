import numpy as np
from tqdm import tqdm
import Xray_winds.Grid_Operations as Grid_Operations

__all__ = ['find_nearest', 'simple_g', 'G', 'projection_2d', 'projection_to_total_lum', 
           'create_list_of_tuples', 'create_spectra']

def find_nearest(array, value):
    """ This function finds the index of the array with the value closest to value.

    Args:
        array (array): The array
        value (int|float): The value you want to find the nearest of

    Returns:
        lowest_idx: tje index of the value in the array closest to value
    """    
    array = np.asarray(array)
    lowest_idx = abs(array - value).argmin()
    return lowest_idx

def simple_g(T, *args, **kwargs):

    return np.where((T >=1e6) * (T <= 1e7), 1, 0)

def G(T, wavelength_band:tuple, which_g='', **kwargs):
    """ Represents the contribution function. If there is no contribution function file found, it will create one. 

    Args:
        T (Array): an array of temperature values
        wavelength_band (tuple): The lower and upper bound of the wavelength band
        which_g (str, optional): If simple it uses the above simple_g function which returns where the temperature of the model is between 1e6 and 1e7. Defaults to ''.

    Returns:
        Array: This array contains the contribution function integrated over the wavelength band.
    """
    from scipy.interpolate import interp1d
    try: 
        if which_g.lower() == 'simple':
            return simple_g(T)
        elif which_g.lower() == 'geom':
            G_total = np.load('G_(T,L)/G-1e-05_geomwvl.npy')
            wvl_array = np.load('G_(T,L)/geom-wvl.npy')
            T_array = np.load('G_(T,L)/temps.npy')
        else:
            # Load in the created G function and the corresponding wvl,T grid
            G_total = np.load('G_(T,L)/G-1e-05.npy')
            wvl_array = np.load('G_(T,L)/wvl.npy')
            T_array = np.load('G_(T,L)/temps.npy')
    except FileNotFoundError:
        print('Creating contribution function')
        import Contribution_function as Contribution
        G_total, wvl_array, T_array = Contribution.create_contribution_function(wavelength_band, T)
    # Find the indicies of the closest wavelengths on the grid
    low_wvl_idx = find_nearest(wvl_array, wavelength_band[0])
    high_wvl_idx = find_nearest(wvl_array, wavelength_band[1])
    G_integrated_across_band = np.trapz(G_total[...,low_wvl_idx:high_wvl_idx],
                                        wvl_array[...,low_wvl_idx:high_wvl_idx], axis=-1)
    
    # Interpolate G for certain wavelength
    T_interpolator = interp1d(T_array, G_integrated_across_band)
    interpolated_fluxes = T_interpolator(T)
    return interpolated_fluxes


def projection_2d(wvl_bin: tuple, stellar_radius: float, interpolator, var_list: list,
                                  image_radius=5, pixel_count=300, angle=(0.,0.), grid_type='linear', segment_size=1/3, *args, **kwargs) -> tuple:
    """ Creates a 2d image of the stellar wind

    Args:
        wvl_bin (tuple): Contains the lower and upper wavelengths between which the projection should be generated
        stellar_radius (float): The radius of the star in cm
        interpolator (Scipy interpolator): Interpolator object that is generated when the data is loaded in with the raw model
        var_list (list): The list of variables available in the model
        image_radius (int, optional): How much of the model the image should contain, i.e the image will be image_radius x image_radius. Defaults to 5.
        pixel_count (int, optional): How pixels should be along one axis. Defaults to 300.
        angle (tuple, optional): Containing the azimuthal and polar angle. Defaults to (0.,0.).
        grid_type (str, optional): The type of grid used, if linear the grid will have the same resolution at each radius. If segmented, the image will have a higher reslution in the center the size of
        image_radius * segment_size. Defaults to 'linear'.
        segment_size (_type_, optional): _description_. Defaults to 1/3.

    Returns:
        tuple: _description_
    """

    image_radius *= stellar_radius
    if float(segment_size) == 1:
        grid_type = 'linear'
        
    if grid_type.lower() == 'segmented':
        # THERE MIGHT BE A FIX FOR THE BORDER OF THE HIGH RES LOW RES PART.
        # We'll create a whole grid and mask out the middle, and create a seperate middle (Primed) grid
        # which is the size of the masked out part with the same resolution
        (X, Y ,Z), _ = Grid_Operations.create_grid(image_radius, pixel_count, type='linear')
        (X_prime, Y_prime, Z_prime), _ = Grid_Operations.create_grid(image_radius * segment_size, pixel_count, type='linear')
        X_rot, Y_rot, Z_rot = Grid_Operations.rotate_grid(angle[0], angle[1], (X, Y ,Z))
        Xprime_rot, Yprime_rot, Zprime_rot = Grid_Operations.rotate_grid(angle[0], angle[1], (X_prime, Y_prime ,Z_prime))

        # The star is only present in the middel grid so we do not have to mask the star in the outer grid
        mask_star = Xprime_rot ** 2 + Yprime_rot ** 2 + Zprime_rot ** 2 <= stellar_radius ** 2
        mask_shadow = (Y_prime ** 2 + Z_prime ** 2 <= stellar_radius ** 2) * (X_prime < 0)
        inner_mask = mask_star + mask_shadow
        
        inner_masked_in_outer =  (X <= image_radius * segment_size) * (Y <= image_radius * segment_size) * (Z <= image_radius * segment_size) \
            * (-image_radius * segment_size <= X) * (-image_radius * segment_size <= Y) * (-image_radius * segment_size <= Z)
        interpolated_inner = interpolator(Xprime_rot, Yprime_rot, Zprime_rot)
        interpolated_outer = interpolator(X_rot, Y_rot, Z_rot)
        integrand_inner = np.square(interpolated_inner[...,var_list.index('Rho [g/cm^3]')] / 1.67e-24) * G(interpolated_inner[...,var_list.index('te [K]')], wvl_bin, *args, **kwargs)
        integrand_outer = np.square(interpolated_outer[...,var_list.index('Rho [g/cm^3]')] / 1.67e-24) * G(interpolated_outer[...,var_list.index('te [K]')], wvl_bin, *args, **kwargs)
        
        masked_integrand_outer = np.where(inner_masked_in_outer==False, integrand_outer, 0)
        masked_integrand_inner = np.where(inner_mask==False, integrand_inner, 0)

        flux_inner = np.trapz(masked_integrand_inner, X_prime, axis=0)
        flux_outer = np.trapz(masked_integrand_outer, X, axis=0)
        return (flux_inner, flux_outer), ((Y_prime[0, :, :], Z_prime[0, :, :]), (Y[0, :, :], Z[0, :, :]))
    
    else:
        (X, Y ,Z), _ = Grid_Operations.create_grid(image_radius, pixel_count, type=grid_type)
        X_rot, Y_rot, Z_rot = Grid_Operations.rotate_grid(angle[0], angle[1], (X, Y ,Z))
        # Interpolate the data on that grid
        interpolated_data = interpolator(X_rot, Y_rot, Z_rot)
        # From the mesh grid create a mask that removes the star
        mask_star = X_rot ** 2 + Y_rot ** 2 + Z_rot ** 2 <= stellar_radius ** 2
        mask_shadow = (Y ** 2 + Z ** 2 <= stellar_radius ** 2) * (X < 0)
        # Mask the shadow of the star
        mask = mask_star + mask_shadow
        integrand = np.square(interpolated_data[...,var_list.index('Rho [g/cm^3]')] / 1.67e-24) * G(interpolated_data[...,var_list.index('te [K]')], wvl_bin, *args, **kwargs)
        masked_integrand = np.where(mask==False, integrand, 0)

        # Now we integrate along the line of sight
        total_flux = np.trapz(masked_integrand, X, axis=0)
        return total_flux, (Y[0, :, :], Z[0, :, :])
    
def projection_to_total_lum(wvl_bin:tuple, stellar_radius, interpolator, *args, **kwargs) -> float:
    """ Converts the projection from projection_2d to a total luminosity

    Args:
        wvl_bin (tuple):  Contains the lower and upper wavelengths between which the projection should be generated
        stellar_radius (_type_): The radius of the star in cm
        interpolator (_type_): Interpolator object that is generated when the data is loaded in with the raw model.

    Returns:
        float: _description_
    """
    LoS_flux, mesh = projection_2d(wvl_bin, stellar_radius, interpolator=interpolator, *args, **kwargs)
    integrate_axis1 = np.trapz(LoS_flux, mesh[1], axis=-1)
    # Then integrate along the other to get a single value
    total_flux_in_band = np.trapz(integrate_axis1, mesh[1][0])
    return total_flux_in_band

def create_list_of_tuples(lst1:list|np.ndarray, lst2:list|np.ndarray) -> list:
    result = []  # Empty list to store the tuples
    for i in range(len(lst1)):
        # Create a tuple from corresponding elements
        tuple_element = (lst1[i], lst2[i])
        result.append(tuple_element)  # Append the tuple to the list
    return result


def create_spectra(wvl_range:tuple, band_width:int|float, disable_tqdm=False, save_spectra=False, **kwargs) -> tuple:
    a = np.arange(wvl_range[0], wvl_range[1],band_width)
    b = np.arange(wvl_range[0] + band_width, wvl_range[1] + band_width, band_width)
    wvl_bands = create_list_of_tuples(a,b)
    center_bands = np.mean(wvl_bands, axis=-1)
    spectrum = np.array([])
    for b in tqdm(wvl_bands, disable=disable_tqdm):
        flux = projection_to_total_lum(b, **kwargs)
        spectrum = np.append(spectrum, flux)
    if save_spectra:
        np.save(save_spectra, [center_bands, spectrum]) # type: ignore
        return (center_bands, spectrum)
    else:
        return (center_bands, spectrum)