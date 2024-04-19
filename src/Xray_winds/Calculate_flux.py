import numpy as np
from tqdm import tqdm
import Xray_winds.src.Xray_winds.Grid_Operations as Grid_Operations

__all__ = ['find_nearest', 'simple_g', 'G', 'projection_2d', 'total_lum_wvl_bin',
           'create_list_of_tuples', 'create_spectra']

def find_nearest(array, value):
    array = np.asarray(array)
    lowest_idx = abs(array - value).argmin()
    return lowest_idx

def simple_g(T, *args, **kwargs):
    return np.where((T >=1e6) * (T <= 1e7), 1, 0)

def G(T, wavelength_band:tuple, which_g='', **kwargs):
    from scipy.interpolate import interp1d
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
                                  image_radius=5, pixel_count=300, angle=(0.,0.), grid_type='linear', *args, **kwargs) -> tuple:

    image_radius *= stellar_radius
    
    if grid_type.lower() == 'segmented':
        # THERE MIGHT BE A FIX FOR THE BORDER OF THE HIGH RES LOW RES PART.
        # We'll create a whole grid and mask out the middle, and create a seperate middle (Primed) grid
        # which is the size of the masked out part with the same resolution
        (X, Y ,Z), _ = Grid_Operations.create_grid(image_radius, pixel_count, type='linear')
        (X_prime, Y_prime, Z_prime), _ = Grid_Operations.create_grid(image_radius/3, pixel_count, type='linear')
        X_rot, Y_rot, Z_rot = Grid_Operations.rotate_grid(angle[0], angle[1], (X, Y ,Z))
        Xprime_rot, Yprime_rot, Zprime_rot = Grid_Operations.rotate_grid(angle[0], angle[1], (X_prime, Y_prime ,Z_prime))

        # The star is only present in the middel grid so we do not have to mask the star in the outer grid
        mask_star = Xprime_rot ** 2 + Yprime_rot ** 2 + Zprime_rot ** 2 <= stellar_radius ** 2
        mask_shadow = (Y_prime ** 2 + Z_prime ** 2 <= stellar_radius ** 2) * (X_prime < 0)
        inner_mask = mask_star + mask_shadow
        
        inner_masked_in_outer =  (X < image_radius/3) * (Y < image_radius/3) * (Z < image_radius/3) \
            * (-image_radius/3 < X) * (-image_radius/3 < Y) * (-image_radius/3 < Z)
        interpolated_inner = interpolator(Xprime_rot, Yprime_rot, Zprime_rot)
        interpolated_outer = interpolator(X_rot, Y_rot, Z_rot)

        integrand_inner = np.square(interpolated_inner[...,var_list.index('Rho [g/cm^3]')] / 1.67e-24) * G(interpolated_inner[...,var_list.index('te [K]')], wvl_bin, *args, **kwargs)
        integrand_outer = np.square(interpolated_outer[...,var_list.index('Rho [g/cm^3]')] / 1.67e-24) * G(interpolated_outer[...,var_list.index('te [K]')], wvl_bin, *args, **kwargs)
        
        masked_integrand_outer = np.where(inner_masked_in_outer==False, integrand_outer, 0)
        masked_integrand_inner = np.where(inner_mask==False, integrand_inner, 0)

        # We want to take into account that light that goes towards the star can't be accounted fors
        Area_of_star = np.square(stellar_radius) * np.pi 
        # Creating an array in the shape of the data to calculate the solid angle of the star at each point
        solid_angle_array_outer = Area_of_star / (X_rot**2 + Y_rot**2 + Z_rot**2)
        solid_angle_array_inner = Area_of_star / (Xprime_rot**2 + Yprime_rot**2 + Zprime_rot**2)

        # The fraction of the sky taken up by the star at a point is then solid_angle / 4pi
        # So the fraction light that escapes is 1 - solid_angle/4pi
        fraction_usable_light_outer = 1 - (solid_angle_array_outer / (4*np.pi))
        fraction_usable_light_inner = 1 - (solid_angle_array_inner / (4*np.pi))

        masked_integrand_inner *= fraction_usable_light_inner
        masked_integrand_outer *= fraction_usable_light_outer

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

        # We want to take into account that light that goes towards the star can't be accounted fors
        Area_of_star = np.square(stellar_radius) * np.pi 
        # Creating an array in the shape of the data to calculate the solid angle of the star at each point
        solid_angle_array = Area_of_star / (X_rot**2 + Y_rot**2 + Z_rot**2)

        # The fraction of the sky taken up by the star at a point is then solid_angle / 4pi
        # So the fraction light that escapes is 1 - solid_angle/4pi
        fraction_usable_light =  1 - (solid_angle_array / (4*np.pi))
        # print(solid_angle_array)
        masked_integrand *= fraction_usable_light

        # Now we integrate along the line of sight
        total_flux = np.trapz(masked_integrand, X, axis=1)
        return total_flux, (Y[:, 0, :], Z[:, 0, :])
    
def projection_to_total_lum(wvl_bin:tuple, stellar_radius, interpolator, *args, **kwargs) -> float:
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