import matplotlib.pyplot as plt
import numpy as np
import Xray_winds.load_data as load_data
import Xray_winds.Grid_Operations as Grid_Operations
import Xray_winds.Calculate_flux as Calculate_flux
import matplotlib as mpl

class star_model():

    def __init__(self, model_name: str, interpolation='nearest', verbose=False):
        self.model_name = model_name
        self.name = model_name.split('x-')[-1]
        self.data = load_data.import_data(self.model_name, interpolate=interpolation, verbose=verbose)
        self.interpolator = self.data[0] 
        self.var_list = self.data[1]
        self.params = self.data[2]

    def raw_data(self):
        """This function returns the 'raw' data from the models.
        """
        return load_data.read_model(self.model_name)
    
    def euclidian_grid(self, size, npix, grid_type='linear'):
        """_summary_

        Args:
            size (_type_): _description_
            npix (_type_): _description_
            grid_type (str, optional): _description_. Defaults to 'linear'.

        Returns:
            _type_: _description_
        """
        (X, Y, Z), _ = Grid_Operations.create_grid(size * self.params['RadiusStar'], npix)
        return (X, Y, Z), self.interpolator(X, Y, Z)
            

    def projection_figure(self, theta, phi, wavelength_range, ax=None, vmax=None, vmin=None, **grid_kw):
        """ Makes a figure with the 2d projection using the projection_2d function of Calculate_flux.py of a stellar wind. The wavelength range, angle, and the vmin-vmax of the colorbar are customizable.
        grid_kw the options of the grid such as the size and resolution are customizable. The grid can be used in segmentation mode, then there will be two pcolormesh objects plotted over eachother. 

        Args:
            theta (float): Azimuthal rotation in radians
            phi (float): Polar rotation angle in radians
            wavelength_range (tuple): Contains the lower and upper wavelengths between which the projection should be generated
            ax (mpl.Axes.axes, optional): The axis on which the projection should be made, if None, a new axes object on which the projection will be made. Defaults to None.
            vmax (float, optional): The minum value of the colorbar, if None the max Irradiance value of the projection will be used. Defaults to None.
            vmin (float, optional): The maximum value of the colorbar, if None vmin = vmax / 1e6. Defaults to None.

        Returns:
            matplotlib.collection.Quadmesh: The quadmesh object of the pcolormesh
        """
        from matplotlib.colors import LogNorm
        if ax == None:
            plt.gca()
        if grid_kw['segment_size'] == 1:
            grid_kw['grid_type'] = 'linear'
        # Give the outer edges less resolution because the structures are bigger
        if grid_kw['grid_type'] == 'segmented':
            lums, grids = Calculate_flux.projection_2d(wavelength_range, self.params['RadiusStar'], self.interpolator, self.var_list, angle=(theta, phi), **grid_kw)
            inner_grid, outer_grid = grids[0], grids[1]
            inner_lum, outer_lum = lums[0], lums[1]
            
            if vmax == None:
                vmax = np.nanmax(lums)
            if vmin == None:
                vmin = vmax / 1e6
            assert vmin < vmax
            norm = LogNorm(vmax=vmax ,vmin=vmin)
            
            outermesh = ax.pcolormesh(outer_grid[0]/self.params['RadiusStar'], outer_grid[1]/self.params['RadiusStar'], outer_lum, norm=norm, shading='gouraud', rasterized=True)
            innermesh = ax.pcolormesh(inner_grid[0]/self.params['RadiusStar'], inner_grid[1]/self.params['RadiusStar'], inner_lum, norm=norm, shading='gouraud', rasterized=True)
            return outermesh

        else:
            lum, mesh = Calculate_flux.projection_2d(wavelength_range, self.params['RadiusStar'], self.interpolator, self.var_list, angle=(theta, phi), **grid_kw)
            mesh = (mesh[0]/self.params['RadiusStar'], mesh[1]/self.params["RadiusStar"])
            if vmax == None:
                vmax = np.nanmax(lum)

            if vmin == None:
                vmin = vmax / 1e6
            norm = LogNorm(vmax=vmax, vmin=vmax/1e6, clip=True)
            quadmesh = ax.pcolormesh(mesh[0], mesh[1], lum, 
                                    norm=norm, rasterized=True)
            return quadmesh
    
    def pop_plot_star(self, theta:float|int, phi:float|int, wavelength_range:tuple, save='', ax=None, **grid_kw) -> None:
        """Creates a pop plot of the star. This pop plot can be saved if a path is given to the 'save'
        variable. The plot is created by interpolating the whole 

        Args:
            theta (float | int): Inclination angle. If 0 (default) the rotation axis is alligned upright
            phi (float | int): Azimuthal angle (East/west rotation)
            wavelength_range (tuple): The wavelengths between the emission is calculated.
            save (str, optional): Defaults to ''. If not an empty string, the path the figure will be saved too.

        return:
            None: Just a popup plot of the star
        """
        from matplotlib.colors import LogNorm
        if ax == None:
            ax = plt.gca()
        if grid_kw['grid_type'] == 'segmented':
            lums, grids = Calculate_flux.projection_2d(wavelength_range, self.params['RadiusStar'], self.interpolator, self.var_list, angle=(theta, phi), **grid_kw)
            inner_grid, outer_grid = grids[0], grids[1]
            inner_lum, outer_lum = lums[0], lums[1]
            vmax = np.nanmax(lums)
            norm = LogNorm(vmax=vmax ,vmin=vmax/1e6)

            outermesh = ax.pcolormesh(outer_grid[0]/self.params['RadiusStar'], outer_grid[1]/self.params['RadiusStar'], outer_lum, norm=norm, shading='gouraud', zorder=0, snap=True)
            innermesh = ax.pcolormesh(inner_grid[0]/self.params['RadiusStar'], inner_grid[1]/self.params['RadiusStar'], inner_lum, norm=norm, shading='gouraud', zorder=1, snap=True)
            
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
        """Generates a spectrum of the stellar wind

        Args:
            wvl_range (tuple): The spectral wavelength range
            spec_res (int | float): The 'Spectral resolution' The amount of bins to describe the spectrum

        Returns:
            tuple: Containing the wavelength array and flux array
        """
        wvl, spectrum = Calculate_flux.create_spectra(wvl_range, spec_res, stellar_radius=self.params['RadiusStar'], interpolator=self.interpolator,
                                                    var_list=self.var_list, **spec_kwargs)
        return wvl, spectrum
    
    def spectrum_pop_plot(self, wvl_range:tuple|str, spec_res:int|float,  **spec_kwargs):
        """Simmilar to 'spectrum' but instead of return makes a popup plot

        Args:
            wvl_range (tuple | str): _description_
            spec_res (int | float): _description_
        """
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
            for n, segment in enumerate(segments):
                X, Y, Z = segment

                star_mask = X ** 2 + Y ** 2 + Z ** 2 <= self.params['RadiusStar'] ** 2
                interpolated_data = self.interpolator(X, Y, Z)
                integrand = np.square(interpolated_data[...,self.var_list.index('Rho [g/cm^3]')] / 1.67e-24) * Calculate_flux.G(interpolated_data[...,self.var_list.index('te [K]')], wvl_bin)

                area_of_star =  np.square(self.params['RadiusStar']) * np.pi
                with np.errstate(divide='ignore'):
                    solid_angle_array = area_of_star / (X ** 2 + Y ** 2 + Z ** 2)
                fraction_light = 1 - (solid_angle_array / (4*np.pi))
                integrand *= fraction_light
                masked_integrand = np.where(star_mask==False, integrand, 0)

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
            one_d = np.trapz(two_d, Y[0,:,:], axis=0)
            self.total_lum= np.trapz(one_d, Z[0, 0, :])
        return self.total_lum
    
    def B_field(self):
        pass
    
    def fit_B_dipole(self):
        pass

    def make_movie(self, duration:float, include_lightcurve:bool, wvl_range=(0.1, 20), name='No_name', grid_kw={}, **lc_kwargs) -> None:
        """Used to make a movie of the star as it rotates around a specified axis. Optionally a lightcurve can be included

        Args:
            duration (float): _description_
            include_lightcurve (bool): _description_
            wvl_range (tuple, optional): _description_. Defaults to (0.1, 20).
            name (str, optional): _description_. Defaults to 'No_name'.
            grid_kw (dict, optional): _description_. Defaults to {}.

        Returns:
            _type_: _description_
        """
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