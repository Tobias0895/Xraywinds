import Xray_winds.load_data as load_data
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import Xray_winds.Calculate_flux as Calculate_flux

class star_model():

    def __init__(self, name, interpolation='nearest', verbose=False):
        self.name = name
        data = load_data.import_data(self.name, interpolate=interpolation, verbose=verbose)
        self.interpolator = data[0] 
        self.var_list = data[1]
        self.params = data[2]

    def raw_data(self):
        return load_data.read_model(self.name)
    
    def projection_figure(self, theta, phi, wavelength_range, **grid_kw):
        from matplotlib.colors import LogNorm
        fig, ax = plt.subplots(1,1)
        # Give the outer edges less resolution because the structures are bigger 
        if grid_kw['grid_type'] == 'segmented':
            lums, grids = Calculate_flux.projection_2d(wavelength_range, self.params['RadiusStar'], self.interpolator, self.var_list, angle=(theta, phi), **grid_kw)
            inner_grid, outer_grid = grids[0], grids[1]
            inner_lum, outer_lum = lums[0], lums[1]
            vmax = np.nanmax(lums)
            norm = LogNorm(vmax=vmax ,vmin=vmax/1e6)


            outermesh = ax.pcolormesh(outer_grid[0]/self.params['RadiusStar'], outer_grid[1]/self.params['RadiusStar'], outer_lum, norm=norm, shading='gouraud')
            innermesh = ax.pcolormesh(inner_grid[0]/self.params['RadiusStar'], inner_grid[1]/self.params['RadiusStar'], inner_lum, norm=norm, shading='gouraud')
            
            ax.set_xlabel('R')
            ax.set_ylabel('Rotation axis')
            fig.colorbar(outermesh)

        else:
            lum, mesh = Calculate_flux.projection_2d(wavelength_range, self.params['RadiusStar'], self.interpolator, self.var_list, angle=(theta, phi), **grid_kw)
            mesh = (mesh[0]/self.params['RadiusStar'], mesh[1]/self.params["RadiusStar"])
            vmax = np.nanmax(lum)
            norm = LogNorm(vmax=vmax, vmin=vmax/1e6, clip=True)
            quadmesh = ax.pcolormesh(mesh[0], mesh[1], lum, 
                                    norm=norm)
            
            fig.colorbar(quadmesh, label='L$_X$ (erg s$^{-1}$ cm$^{-2}$)')
            ax.set_xlabel('R (r$_{star}$)')
            ax.set_ylabel('R (r$_{star}$)')

        return fig, ax
    def pop_plot_star(self, theta:float|int, phi:float|int, wavelength_range:tuple, save='', **grid_kw) -> None:
        """Creates a pop plot of the star. This pop plot can be saved if a path is given to the 'save'
        variable. The plot is created by interpolating the whole 

        Args:
            theta (float | int): Inclination angle. If 0 (default) the rotation axis is alligned upright
            phi (float | int): Azimuthal angle (East/west rotation)
            wavelength_range (tuple): The wavelengths between the emission is calculated.
            save (str, optional): Defaults to ''. If not an empty string, the path the figure will be saved too.
        """
        from matplotlib.colors import LogNorm
        if grid_kw['grid_type'] == 'segmented':
            lums, grids = Calculate_flux.projection_2d(wavelength_range, self.params['RadiusStar'], self.interpolator, self.var_list, angle=(theta, phi), **grid_kw)
            inner_grid, outer_grid = grids[0], grids[1]
            inner_lum, outer_lum = lums[0], lums[1]
            vmax = np.nanmax(lums)
            norm = LogNorm(vmax=vmax ,vmin=vmax/1e6)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            outermesh = ax.pcolormesh(outer_grid[0]/self.params['RadiusStar'], outer_grid[1]/self.params['RadiusStar'], outer_lum, norm=norm, shading='gouraud', zorder=0, snap=True)
            innermesh = ax.pcolormesh(inner_grid[0]/self.params['RadiusStar'], inner_grid[1]/self.params['RadiusStar'], inner_lum, norm=norm, shading='gouraud', zorder=1, snap=True)
            

            fig.colorbar(innermesh)
            ax.set_xlabel('R')
            ax.set_ylabel('Rotation axis')

        else:
            lum, mesh = Calculate_flux.projection_2d(wavelength_range, self.params['RadiusStar'], self.interpolator, self.var_list, angle=(theta, phi), **grid_kw)
            mesh = (mesh[0]/self.params['RadiusStar'], mesh[1]/self.params["RadiusStar"])
            vmax = np.nanmax(lum)
            norm = LogNorm(vmax=vmax, vmin=vmax/1e6, clip=True)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            quadmesh = ax.pcolormesh(mesh[0], mesh[1], lum, 
                                    norm=norm, zorder=0)
            
            fig.colorbar(quadmesh, label='L$_X$ (erg s$^{-1}$ cm$^{-2}$)')
            ax.set_xlabel('R (r$_{star}$)')
            ax.set_ylabel('R (r$_{star}$)')
        
        if save:
            plt.savefig(save, dpi=500, bbox_inches='tight')
            plt.show(block=False)
        else:
            plt.show()

    def spectrum(self, wvl_range:tuple, spec_res:int|float, **spec_kwargs) -> tuple:
        wvl, spectrum = Calculate_flux.create_spectra(wvl_range, spec_res, stellar_radius=self.params['RadiusStar'], interpolator=self.interpolator,
                                                    var_list=self.var_list, **spec_kwargs)
        return wvl, spectrum
    
    def spectrum_pop_plot(self, wvl_range:tuple|str, spec_res:int|float,  **spec_kwargs):
        if wvl_range.lower() == 'complete': # type: igonore
            wvl, spec = self.spectrum((0.1, 180), spec_res, **spec_kwargs)
        else:
            wvl, spec = self.spectrum(wvl_range, spec_res, **spec_kwargs) # type: ignore
        spec_fig = plt.figure()
        ax = spec_fig.add_subplot(111)
        ax.plot(wvl, spec)
        ax.set_yscale('log')
        ax.set_xlabel('Wavelength ($\AA$)')
        ax.set_ylabel('L$_X$ (erg s$^{-1}$ $\AA^{-1}$)')
        plt.show()

    def light_curve(self, rotation_direction:str, wvl_range=(0.1, 180), inclination=0, disable_tqdm=False, steps=50, **kwargs):
        from tqdm import tqdm
        angles = np.linspace(0, 2*np.pi, steps) # in radians
        fluxes = np.zeros(len(angles))
        if rotation_direction.lower() == 'equator':
            for i, ang in enumerate(tqdm(angles, disable=disable_tqdm)):
                fluxes[i] = Calculate_flux.projection_to_total_lum(wvl_range, stellar_radius=self.params['RadiusStar'], interpolator=self.interpolator,
                                                              var_list=self.var_list, angle=(ang, inclination))
        elif rotation_direction.lower() == 'poles':
            for i, ang in enumerate(tqdm(angles)):
                fluxes[i] = Calculate_flux.projection_to_total_lum(wvl_range, stellar_radius=self.params['RadiusStar'], interpolator=self.interpolator, 
                                                             var_list=self.var_list, angle=(0., ang))
        return angles, fluxes
    
    def lum_x(self, image_radius=20, pixel_count=200, wvl_bin=(0.1,180), grid_type='linear', nseg=2, *args, **kwargs):
        import Xray_winds.Grid_Operations as Grid_Operations
        if grid_type =='segmented':
            grid, x= Grid_Operations.create_grid(image_radius * self.params['RadiusStar'], pixel_count, 'linear')
            segments = Grid_Operations.up_center_res(grid)
            if nseg > 1: # If we want to segment the grid more then once
                i = 1
                while i < nseg:
                    middle = segments.pop(-1)
                    segments_inner = Grid_Operations.up_center_res(middle)
                    segments += segments_inner
                    i+=1
            
            total_X = []
            for segment in segments:
                X, Y, Z = segment

                star_mask = X ** 2 + Y ** 2 + Z ** 2 <= self.params['RadiusStar'] ** 2
                interpolated_data = self.interpolator(X, Y, Z)
                integrand = np.square(interpolated_data[...,self.var_list.index('Rho [g/cm^3]')] / 1.67e-24) * Calculate_flux.G(interpolated_data[...,self.var_list.index('te [K]')], wvl_bin)
                masked_integrand = np.where(star_mask==False, integrand, 0)

                area_of_star =  np.square(self.params['RadiusStar']) * np.pi
                solid_angle_array = area_of_star / (X ** 2 + Y ** 2 + Z ** 2)
                fraction_light = 1 - (solid_angle_array / (4*np.pi))
                masked_integrand *= fraction_light

                projection = np.trapz(masked_integrand, X, axis=0)

                one_d = np.trapz(projection, Y[0,:,:], axis=0)
                tot = np.trapz(one_d, Z[0, 0, :])
                total_X.append(tot)

            return np.sum(total_X)
        else:
            # We first create a 3D meshgrid
            image_radius *= self.params['RadiusStar']
            (X, Y ,Z), _ = Grid_Operations.create_grid(image_radius, pixel_count, type=grid_type)
            # Interpolate the data on that grid
            interpolated_data = self.interpolator(X, Y, Z)
            # From the mesh grid create a mask that removes the star
            mask = X ** 2 + Y ** 2 + Z ** 2 <= self.params['RadiusStar'] ** 2

            integrand = np.square(interpolated_data[...,self.var_list.index('Rho [g/cm^3]')] / 1.67e-24) * Calculate_flux.G(interpolated_data[...,self.var_list.index('te [K]')], wvl_bin, *args, **kwargs)
            masked_integrand = np.where(mask==False, integrand, 0)

            # We want to take into account that light that goes towards the star can't be accounted fors
            Area_of_star = np.square(self.params['RadiusStar']) * np.pi 
            # Creating an array in the shape of the data to calculate the solid angle of the star at each point
            solid_angle_array = Area_of_star / (X**2 + Y**2 + Z**2)

            # The fraction of the sky taken up by the star at a point is then solid_angle / 4pi
            # So the fraction light that escapes is 1 - solid_angle/4pi
            fraction_usable_light =  1 - (solid_angle_array / (4*np.pi))
            masked_integrand *= fraction_usable_light
            two_d = np.trapz(masked_integrand, X, axis=0)
            one_d = np.trapz(two_d, Y[0,:,:], axis=-1)
            self.total_lum= np.trapz(one_d, Z[0, 0, :])
        return self.total_lum
    
    def B_field(self):
        pass
    
    def fit_B_dipole(self):
        pass

    def make_movie(self, duration:float, include_lightcurve:bool, wvl_range=(0.1, 20), name='No_name', grid_kw={}, **lc_kwargs) -> None:
        import make_movie
        from moviepy.editor import VideoClip
        from moviepy.video.io.bindings import mplfig_to_npimage

        def make_rotation_lightcurve_frame(t, **kwargs):
            from matplotlib.colors import LogNorm
            fig = plt.figure()
            angle = t * (360/duration)
            fig, ax = plt.subplots(2, 1, figsize=(5,7), gridspec_kw={'height_ratios':[3,1]})
            ax[0].clear()
            ax[1].clear()
            img, mesh = Calculate_flux.projection_2d(wvl_range, direction='+x', stellar_radius=self.params['RadiusStar'], interpolator=self.interpolator,
                                                            angle=(np.deg2rad(angle),0), var_list=self.var_list, use_simple_g=False, **grid_kw)
            vmax = np.nanmax(img)
            # norm = LogNorm(vmax=vmax, vmin=vmax/1e6 , clip=True)
            color = ax[0].pcolormesh(mesh[0]/self.params['RadiusStar'], mesh[1]/self.params['RadiusStar'], img, norm='log')
            ax[0].set_title('Angle = {:.1f} degrees'.format(angle))
            ax[0].axis('equal')
            fig.colorbar(color)
            ax[0].set_ylabel('Z [R]')
            ax[0].set_xlabel('d [R]')

            lum_at_angle = np.interp(angle, np.rad2deg(angles), lum)
            ax[1].plot(np.rad2deg(angles), lum, c='k')
            ax[1].scatter(angle, lum_at_angle, ec='k')
            ax[1].set_xlabel('Angle (rad)')
            ax[1].set_ylabel("L_x erg s^-1")

            return mplfig_to_npimage(fig)


        def make_rotation_frame(t):
            from matplotlib.colors import LogNorm
            fig = plt.figure()
            angle = t * 36
            fig, ax = plt.subplots(1, 1)
            ax.clear()
            img, mesh = Calculate_flux.projection_2d(wvl_range, stellar_radius=self.params['RadiusStar'], interpolator=self.interpolator,
                                                            angle=(np.deg2rad(angle), 0), var_list=self.var_list, use_simple_g=False, **grid_kw)
            vmax = np.nanmax(img)
            norm = LogNorm(vmax=vmax, vmin=vmax/1e6 , clip=True)
            color = ax.pcolormesh(mesh[0]/self.params['RadiusStar'], mesh[1]/self.params['RadiusStar'], img, norm=norm, shading='gouraud')
            ax.set_title('Angle = {:.1f}'.format(angle))
            ax.axis('equal')
            fig.colorbar(color)
            ax.set_ylabel('Z [R]')
            ax.set_xlabel('d [R]')

            return mplfig_to_npimage(fig)


        if include_lightcurve:
            angles, lum = self.light_curve('equator', wvl_range, **lc_kwargs)
            movie = VideoClip(make_rotation_lightcurve_frame, duration=duration)
            movie.write_gif(f'Movies/{name}.gif', fps=15)
        else:
            movie = VideoClip(make_rotation_frame, duration=duration)
            movie.write_gif(f"Movies/{name}.gif", fps=15)


if __name__ == "__main__":
    name = '1x-MEL25-005'
    star = star_model(name)
    star.make_movie(5, True, (0.1, 20), name=f'{name}-Rotation+lightcurve', grid_kw={'pixel_count': 250, 'image_radius': 3})