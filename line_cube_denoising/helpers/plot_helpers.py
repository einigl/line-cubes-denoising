""" Plot helpers """

from multiprocessing.sharedctypes import Value
import os

import subprocess
import shutil
import pickle
import fnmatch

from math import ceil, prod
from typing import Optional, Union, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import cubes_api
from cubes_api import Cube, Map, Profile

from . import distributions_helpers as dist

def catch(msg : Optional[str] = None) :
    """ Decorator for functions in which errors must be catched """
    def decorator(func) :
        def wrapper(*args, **kwargs) :
            try :
                func(*args, **kwargs)
            except KeyboardInterrupt :
                raise
            except :
                if msg is None :
                    print('ERROR CATCHED')
                else :
                    print(f'ERROR CATCHED : {msg}')
        return wrapper
    return decorator

class PlotHelpers :
    """ Implements helpers to plot the results of the regression """

    def __init__(self, path, year, line) :
        """ Initializer """
        self.year = year
        self.line = line

        self.path = path
        self.data_path = os.path.join(self.path, 'data') # TODO
        self.figure_path = os.path.join(self.path, 'figures')

    def create_directory(self) :
        """ Create a figure directory """
        if os.path.isdir(self.figure_path) :
            print('Figures directory already exists. Overwrite.')
            shutil.rmtree(self.figure_path)
        os.makedirs(self.figure_path)
        print('Figures directory created')
    
    ### Plot methods ###

    def _savefig(self, fig, path : str, filename : str) :
        """ Helper to save a matplotlib figure even if the directory currently don't exist """
        if not os.path.isdir(path) :
            os.makedirs(path)
        ext = '.png'
        for ext_ in ('.png', '.jpg', '.svg', '.jpeg') :
            if filename.endswith('.png') :
                filename = filename[:-len(ext_)]
                ext = ext_
            break
        
        idx = None
        num = ''
        while os.path.isfile(os.path.join(path, filename + num + ext)) :
            if idx is None :
                idx = 1
            else :
                idx += 1
            num = '_{}'.format(idx)
        fig.savefig( os.path.join(path, filename + num + ext) )
        plt.close(fig)

        print( '{} created'.format(filename + num + ext) )

    def _create_dir_if_doesnt_exist(self, path) :
        """ Helper to create a directory only if he doesn't exist """
        if not os.path.exists(path) or not os.path.isdir(path) :
            os.makedirs(path)

    def _get_fits_in_dir(self, path) :
        """ Returns a list of all FITS in directory path (without the extension) """
        return [f.split('.')[0] for f in os.listdir(path) if f.endswith('.fits')]

    def _figsize(self, rows : int, cols : int) :
        """ Returns the optimal grid size for a subplot in function of the data year """
        if self.year == 2014 :
            return (cols*6.4, rows*4.8)
        if self.year == 2020 :
            return (cols*6.4, rows*6.4)
        raise ValueError('Year must be 2014 or 2020, not {}'.format(self.year))

    @catch()
    def plot_losses(self, log = True) :
        """
        Plot evolution of training and validation losses.
        
        Parameters
        ----------
        log : bool, optional
            If log is True, the y-axis scale is logarithmic. Else, the y-axis scale is linear.

        figDir : str, optional
            Path of folder where the figure has to be saved. If figDir is None, the figure is not saved.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Output matplotlib figure
        """
        # Hard written shape of subplots given number of subplots
        figsizes = [None, (6.4,4.8), (6.4*2,4.8), (6.4*3,4.8), (6.4*2,4.8*2), (6.4*3,4.8*2), (6.4*3,4.8*2)]
        subplotShapes = [None, (1,1), (1,2), (1,3), (2,2), (2,3), (2,3)]

        # Load saved losses
        with open(os.path.join(self.data_path, 'losses.pickle'), 'rb') as f :
            data = pickle.load(f)
            n = len(data)

        # Create subplots
        fig, axs = plt.subplots(subplotShapes[n][0], subplotShapes[n][1], figsize = figsizes[n], squeeze = False)
        axs = axs.flatten()

        # Plot each loss
        for i, name in enumerate(data) :
            if log :
                axs[i].semilogy( data[name]['train'], color = 'b', label = 'train' )
                axs[i].semilogy( data[name]['val'], color = 'r', label = 'test' )
            else :
                axs[i].plot( data[name]['train'], color = 'b', label = 'train' )
                axs[i].plot( data[name]['val'], color = 'r', label = 'test' )
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel('Loss')
            axs[i].set_title(name)
            axs[i].legend(loc = 'upper right')
        for i in range(len(data), prod(subplotShapes[n])) :
            axs[i].set_visible(False)

        # Save figure
        fig.tight_layout()
        self._savefig( fig, self.figure_path, 'losses.png')

    @catch()
    def plot_dataset_splitting(self) :
        """
        Plot a representation of the dataset splitting.

        Parameters
        ----------

        Returns
        -------
        """
        print('Plotting dataset splitting')
    
        cubes_api.from_fits(Map, 'splitting_map', self.data_path).save_plot('splitting_map', self.figure_path)

    @catch()
    def plot_dataset_balancing(self) :
        """
        Plot duplication weights to balance the dataset.
        
        Parameters
        ----------

        Returns
        -------
        """
        print('Plotting dataset rebalancing statistics')

        # TODO

    @catch()
    def plot_noise_map(self) :
        """ Plot channels of cube """
        noise_map = cubes_api.from_fits( Map, self.line, os.path.join(self.data_path, 'noise_maps') )

        fig, ax = plt.subplots(1, 1)
        noise_map.plot(ax = ax, vmin = 0)

        # Save plot
        self._savefig(fig, os.path.join(self.figure_path, 'noise_maps'), self.line)

    @catch()
    def plot_bottleneck(self, norm : bool = True) :
        """ Plot and save one by one the bottleneck of the network """
        files_list = os.listdir( os.path.join(self.data_path, 'bottleneck') )
        files_list = fnmatch.filter(files_list, self.line + '_*.fits')
        for file in files_list :
            map = cubes_api.from_fits( Map, file, os.path.join(self.data_path, 'bottleneck') )
            if norm :
                map = (map - map.min(float)) / (map.max(float) - map.min(float))

            # Create plot
            fig = plt.figure()
            map.plot()

            # Save plot
            self._savefig(fig, os.path.join(self.figure_path, 'bottleneck'), file.replace('.fits', '.png'))

    @catch()
    def plot_cube(self, channels : list, grid : tuple, filename : str, folder : Optional[str] = None, **kwargs) :
        """ Plot channels of cube """
        folder = '' if folder is None else folder
        cube = cubes_api.from_fits( Cube, filename, os.path.join(self.data_path, folder) )
        if len(channels) > grid[0] * grid[1] :
            raise ValueError(f'Too many channels for given grid (max {grid[0] * grid[1]})')

        fig, axs = plt.subplots(grid[0], grid[1], figsize = self._figsize(grid[0], grid[1]), squeeze = False)
        axs = axs.flatten()

        for i, k in enumerate(channels) :
            cube.plot_channel(k, ax = axs[i], no_logical = True, **kwargs)
        for i in range(len(channels), grid[0]*grid[1]) :
            axs[k].set_visible(False)

        # Save plot
        self._savefig( fig, os.path.join(self.figure_path, folder),\
                '{}.png'.format(filename.split('.')[0]) )

    #@catch()
    def plot_images_comparison(self, channels) :
        """ TODO """
        inputs = cubes_api.from_fits( Cube, self.line, os.path.join(self.data_path, 'inputs') )
        outputs = cubes_api.from_fits( Cube, self.line, os.path.join(self.data_path, 'outputs') )
        residues = cubes_api.from_fits( Cube, self.line, os.path.join(self.data_path, 'residues') )
        noise_map = cubes_api.from_fits( Map, self.line, os.path.join(self.data_path, 'noise_maps') )

        norm_residues = residues / noise_map

        if channels.lower() == 'all' :
            channels = list(range(inputs.shape[0]))

        if self.year == 2014 :
            rows = 4 # Can be modified
            figsize = (3*6.4, rows*4.8)
        elif self.year == 2020 :
            rows = 4 # Can be modified
            figsize = (3*6.4, rows*6.4)
        else :
            raise ValueError('Year must be 2014 or 2020, not {}'.format(self.year))

        row = 0
        for k in channels :
            if row == 0 :
                fig, axs = plt.subplots(rows, 4, figsize = figsize, dpi = 200)
            vmin = 0
            vmax = max( inputs.get_channel(k).max(float), outputs.get_channel(k).max(float) )
            inputs.plot_channel(k, ax = axs[row, 0], vmin = vmin, vmax = vmax, no_logical = True)
            outputs.plot_channel(k, ax = axs[row, 1], vmin = vmin, vmax = vmax + 1e-2, no_logical = True)
            vmin = -residues.get_channel(k).abs().max(float) #-np.nanpercentile(np.abs(residues)[k], 95)
            vmax = residues.get_channel(k).abs().max(float) #np.nanpercentile(np.abs(residues)[k], 95)
            residues.plot_channel(k, ax = axs[row, 2], vmin = vmin, vmax = vmax, no_logical = True)
            vmin = -5
            vmax = 5
            norm_residues.plot_channel(k, ax = axs[row, 3], vmin = vmin, vmax = vmax, no_logical = True)
            fig.tight_layout()
            row += 1
            if row >= rows and k != channels[-1] :
                self._savefig( fig, os.path.join(self.figure_path, 'outputs'),\
                    '{}_images_comp.png'.format(self.line) )
                row = 0
        for row_ in range(row, rows) :
            axs[row_, 0].set_visible(False)
            axs[row_, 1].set_visible(False)
            axs[row_, 2].set_visible(False)
            axs[row_, 3].set_visible(False)
        if row != 0 :
            self._savefig( fig, os.path.join(self.figure_path, 'outputs'),\
                '{}_images_comp.png'.format(self.line) )

    @catch()
    def plot_histo2d(self, log : bool = False) :
        """ TODO """
        inputs = cubes_api.from_fits( Cube, self.line, os.path.join(self.data_path, 'inputs') )
        outputs = cubes_api.from_fits( Cube, self.line, os.path.join(self.data_path, 'outputs') )

        a = inputs.data.flatten()
        a = a[~np.isnan(a)]
        b = outputs.data.flatten()
        b = b[~np.isnan(b)]
        
        if log :
            vmin = 1e-1
            a, b = np.log10(a[(a > vmin) & (b > vmin)]), np.log10(b[(a > vmin) & (b > vmin)])

        ab_min = min(a.min(), b.min())
        ab_max = max(a.max(), b.max())

        H, xedges, yedges = np.histogram2d(a, b, bins = 100, density = True,
            range = [[ab_min, ab_max], [ab_min, ab_max]])
        X, Y = np.meshgrid(xedges, yedges)

        fig, ax = plt.subplots(1, 1, figsize = (6.4 * 1.5, 4.8 * 1.5))
        im = ax.pcolormesh(X, Y, H, norm = LogNorm(), cmap = 'jet')
        fig.colorbar(im, ax = ax)
        ax.plot([xedges[0], xedges[-1]], [xedges[0], xedges[-1]], '--k')

        # Save histograms
        if log :
            self._savefig( fig, os.path.join(self.figure_path, 'outputs'),
                '{}_histo2d_log.png'.format(self.line))
        else :
            self._savefig( fig, os.path.join(self.figure_path, 'outputs'),
                '{}_histo2d.png'.format(self.line))

    @catch()
    def plot_residues_distribution(self, log = False) :
        """ Plot the distribution of the residues compared to target N(0,1) distribution """
        noise_map = cubes_api.from_fits( Map, self.line, os.path.join(self.data_path, 'noise_maps') )
        residues = cubes_api.from_fits( Cube, self.line, os.path.join(self.data_path, 'residues') )
        inputs = cubes_api.from_fits( Cube, self.line, os.path.join(self.data_path, 'inputs') )

        # Normalized residues
        residues /= noise_map

        # Integrated inputs
        inputs_map = inputs.integral().data

        # Target distribution
        gaussian = lambda t : 1 / (2*np.pi)**0.5 * np.exp( -t**2/2 )

        # 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize = [6.4 * 2, 4.8 * 2])
        axs = axs.flatten()
        nPoints = 1000
        nBins = 40
        a, b = -10, 10

        # Plot helper
        def _helper(ax, data) :
            _data = data[(data >= a) & (data <= b)]
            heights, _, __ = ax.hist(_data.flatten(), bins = nBins, density = True)
            t = np.linspace(a, b, nPoints)
            ax2 = ax.twinx()
            ax2.plot(t, gaussian(t), 'r')
            ax.set_xlabel('Residues')
            ax.set_ylabel('Output residues dentity')
            ax2.set_ylabel('Target density')
            if log :
                ax.set_xlim([a, b])
                ax.set_ylim([1e-8, 1.5 * max(np.max(heights), gaussian(0))])
                ax2.set_ylim([1e-8, 1.5 * max(np.max(heights), gaussian(0))])
                ax.set_yscale('log')
                ax2.set_yscale('log')
            else :
                ax.set_xlim([a, b])
                ax.set_ylim([0, 0.05 + max(np.max(heights), gaussian(0))])
                ax2.set_ylim([0, 0.05 + max(np.max(heights), gaussian(0))])

        # Distribution of all pixels
        _helper(axs[0], residues.data)
        plt.title('All pixels')

        # Distribution of pixels of the first quartile
        selected_pixels = inputs_map < np.percentile(inputs_map, 25)
        _helper(axs[1], residues.data[:, selected_pixels])
        plt.title('Only 25% pixels with the lowest integrated value')

        # Distribution of pixels of the last quartile
        selected_pixels = inputs_map > np.percentile(inputs_map, 75)
        _helper(axs[2], residues.data[:, selected_pixels])
        plt.title('Only 25% pixels with the highest integrated value')

        # Distribution of pixels of the last decile
        selected_pixels = inputs_map > np.percentile(inputs_map, 90)
        _helper(axs[3], residues.data[:, selected_pixels])
        plt.title('Only 10% pixels with the highest integrated value')

        # Save histograms
        fig.tight_layout()
        if log :
            self._savefig( fig, os.path.join(self.figure_path, 'noise_analysis'), 'residues_distribution_log.png')
        else :
            self._savefig( fig, os.path.join(self.figure_path, 'noise_analysis'), 'residues_distribution.png')

    @catch()
    def plot_residues_map(self) :
        """ Plot a map of the residues RMS by pixel """
        noise_map = cubes_api.from_fits( Map, self.line, os.path.join(self.data_path, 'noise_maps') )
        residues = cubes_api.from_fits( Cube, self.line, os.path.join(self.data_path, 'residues') )

        # RMS map
        rms_map = residues.std(Map)
        vmax = max( noise_map.max(float), rms_map.max(float) )
        
        fig = plt.figure(figsize = [2 * 6.4, 2 * 4.8])

        # Plot residues map
        plt.subplot(2, 2, 1)
        rms_map.plot(vmin = 0, vmax = vmax)
        plt.title('RMS of residues')

        # Plot noise map
        plt.subplot(2, 2, 2)
        noise_map.plot(vmin = 0, vmax = vmax)
        plt.title('RMS of noise')

        # Plot difference
        plt.subplot(2, 2, 3)
        (rms_map - noise_map).plot()
        plt.title('Difference of RMS')

        # Plot log ratio
        plt.subplot(2, 2, 4)
        (rms_map / noise_map).plot(norm = LogNorm(5e-1, 5))
        plt.title('Ratio of RMS')

        # Save map
        plt.tight_layout()
        self._savefig( fig, os.path.join(self.figure_path, 'noise_analysis'), 'residues_map.png')

    @catch()
    def plot_kullback(self) :
        """ Plot a map of the pixel-wise KL divergence """
        noise_map = Map.from_fits( self.line, os.path.join(self.data_path, 'noise_maps') )
        residues = Cube.from_fits( self.line, os.path.join(self.data_path, 'residues') )

        norm_residues = residues / noise_map
        t, kde = norm_residues.kde(axis = 'spectral', t_step = 0.1)
        dt = t[1] - t[0]

        target = dist.normal_pdf(t)

        distance_map : Map = Map.from_numpy(dist.kld(kde, target, dt), noise_map.header)

        fig = plt.figure()
        distance_map.plot()
        plt.title('KL divergence of residues (average : {:.2f})'.format(distance_map.mean()))

        # Save the figure
        self._savefig(fig, os.path.join(self.figure_path, 'noise_analysis'), 'kl_divergence.png')

    @catch()
    def plot_residues_distance(self) :
        """ Plot a map of the euclidean distance between normalized residues and N(0,1) """
        noise_map = Map.from_fits( self.line, os.path.join(self.data_path, 'noise_maps') )
        residues = Cube.from_fits( self.line, os.path.join(self.data_path, 'residues') )

        # Normalized residues
        residues = (residues / noise_map).data

        # Euclidean distance for each pixel between normalized residues and N(0,1)
        h = 1 / residues.nz**(1/5)
        ones = np.ones_like(residues[:, 0, 0])
        distance_map = np.zeros(noise_map.shape)
        for i in range(distance_map.shape[0]) :
            for j in range(distance_map.shape[1]) :
                x = residues[:, i, j]
                grid = np.outer(x, ones)
                distance_map[i, j] = 1/(2*np.pi**0.5) * (
                    1/h * np.mean( np.exp( -1/(4*h**2) * (grid-grid.T)**2 ) )
                    - (8/(1+h**2))**0.5 * np.mean( np.exp( -x**2/(2*(1+h**2)) ) )
                    + 1
                )
        
        # Plot the map
        fig = plt.figure()
        plt.imshow( distance_map, cmap = 'jet', norm = LogNorm(vmin=0.001, vmax=np.max(distance_map)) )
        plt.colorbar()

        # Save the figure
        self._savefig(fig, os.path.join(self.figure_path, 'noise_analysis'), 'residues_distance.png')

    @catch()
    def plot_kde(self, channels : Union[Sequence, str]) :
        """ TODO """
        noise_map = cubes_api.from_fits( Map, self.line, os.path.join(self.data_path, 'noise_maps') )
        residues = cubes_api.from_fits( Cube, self.line, os.path.join(self.data_path, 'residues') )

        norm_residues = (residues / noise_map)

        if channels.lower() == 'all' :
            channels = list(range(residues.nz))

        cols = 4 # Can be modified
        rows = 2

        a, b = -5, 5 # Can be modified
        dt = 0.05 # Can be modified

        num = 0
        for i, k in enumerate(channels) :
            if num == 0 :
                fig, axs = plt.subplots(rows, cols, figsize = (1.5 * cols/1.3 * 6.4, 1.5 * rows * 4.8))
                axs = axs.flatten()
            t, pdf = norm_residues.get_channel(k).kde(t_bounds = (a, b), t_step = dt)
            if i == 0 :
                target = dist.normal_pdf(t)
            axs[num].plot(t, target, '--', linewidth = 2)
            axs[num].plot(t, pdf, linewidth = 2)
            axs[num].set_title(f'Channel {k+1}')
            num += 1
            if num >= rows*cols and k != channels[-1] :
                self._savefig(fig, os.path.join(self.figure_path, 'noise_analysis'), 'densities.png')
                num = 0
        for num_ in range(num, rows*cols) :
            axs[num_].set_visible(False)
        if num != 0 :
            self._savefig(fig, os.path.join(self.figure_path, 'noise_analysis'), 'densities.png')
    
    @catch()
    def plot_criteria(self) :
        """ TODO """
        noise_map = Map.from_fits( self.line, os.path.join(self.data_path, 'noise_maps') )
        residues = Cube.from_fits( self.line, os.path.join(self.data_path, 'residues') )

        norm_residues = (residues / noise_map).to_numpy('pixel')

        fig, axs = plt.subplots(3, 2, figsize = (3 * 6.4, 3 * 4.8))
        axs = axs.flatten()

        a, b = -5, 5 # Can be modified
        dt = 0.05 # Can be modified

        t, target = dist.normal_pdf(a, b, dt)
        _, pdfs = dist.kde(norm_residues.T, a, b, dt)

        axs[0].plot( np.mean(residues.to_numpy('pixel'), axis = 0), linewidth = 2, color = 'orange' )
        axs[0].set_xlabel("Velocity channel")
        axs[0].set_ylabel("Criterion")
        axs[0].set_title("$\Phi(residues)$ (optimal is 0, worst is $\pm\infty$)")

        axs[1].plot( dist.l2(pdfs, target, dt), linewidth = 2, color = 'orange' )
        axs[1].set_xlabel("Velocity channel")
        axs[1].set_ylabel("Criterion")
        axs[1].set_title("$||residues-target||_2$ (optimal is 0, worst is 1)")

        axs[2].plot( dist.kld(pdfs, target, dt), linewidth = 2, color = 'orange' )
        axs[2].set_xlabel("Velocity channel")
        axs[2].set_ylabel("Criterion")
        axs[2].set_title("$KLD(residues||target)$ (optimal is 0, worst is $\infty$)")

        axs[3].plot( dist.kld(target, pdfs, dt), linewidth = 2, color = 'orange' )
        axs[3].set_xlabel("Velocity channel")
        axs[3].set_ylabel("Criterion")
        axs[3].set_title("$KLD(target||residues)$ (optimal is 0, worst is $\infty$)")

        axs[4].plot( dist.jsd(pdfs, target, dt, base = 2), linewidth = 2, color = 'orange' )
        axs[4].set_xlabel("Velocity channel")
        axs[4].set_ylabel("Criterion")
        axs[4].set_title("$\sqrt{JSD(residues||target)}$ (optimal is 0, worst is 1)")

        n_sigma = 3
        axs[5].plot( dist.outliers_ratio(t, pdfs, target, dt, n_sigma = n_sigma), linewidth = 2, color = 'orange' )
        axs[5].set_xlabel("Velocity channel")
        axs[5].set_ylabel("Criterion")
        axs[5].set_title(f"{n_sigma}-sigma ratio (optimal is 1, worst is 0 or $\infty$)")

        fig.tight_layout()

        # Save the figure
        self._savefig(fig, os.path.join(self.figure_path, 'noise_analysis'), 'criteria.png')

    @catch()
    def plot_profiles(self, nProfiles : int, coord : list = None, seed : int = 0) :
        """ On `nProfiles` subplots, plot a residues profile """
        inputs = cubes_api.from_fits( Cube, self.line, os.path.join(self.data_path, 'inputs') )
        outputs = cubes_api.from_fits( Cube, self.line, os.path.join(self.data_path, 'outputs') )

        # Create 2 x n subplots
        rows = ceil(nProfiles/2)
        fig, axs = plt.subplots(rows, 2, figsize = (2 * 6.4, rows//2 * 4.8))
        axs = axs.flatten()
        np.random.seed(seed)

        # Generate coordinates in None and plot profiles
        for i in range(nProfiles) :
            if coord is None or len(coord) <= i :
                x = np.random.randint(inputs.shape[2])
                y = np.random.randint(inputs.shape[1])
            else :
                x,y = coord[i]
            inputs.plot_pixel((x,y), ax = axs[i], label = 'Input', linestyle = '--')
            outputs.plot_pixel((x,y), ax = axs[i], label = 'Output', linestyle = '-')
            axs[i].set_title('Pixel ({},{})'.format(x,y))
            axs[i].legend(loc = 'upper right')
        if nProfiles%2 == 1 :
            axs[-1].set_visible(False)

        # Save figure
        fig.tight_layout()
        self._savefig(fig, os.path.join(self.figure_path, 'outputs'), 'profiles.png')

    @catch()
    def plot_residues_profiles(self, nProfiles : int, coord : list = None, seed : int = 0) :
        """ On `nProfiles` subplots, plot a residues profile """
        noise_map = cubes_api.from_fits( Map, self.line, os.path.join(self.data_path, 'noise_maps') )
        residues = cubes_api.from_fits( Cube, self.line, os.path.join(self.data_path, 'residues') )

        # Create 2 x n subplots
        rows = ceil(nProfiles/2)
        fig, axs = plt.subplots(rows, 2, figsize = (2 * 6.4, rows//2 * 4.8))
        axs = axs.flatten()
        np.random.seed(seed)

        # Generate coordinates in None and plot profiles
        """one_profile = cubes.ones_like(residues.get_pixels((0,0)))"""
        for i in range(nProfiles) :
            if coord is None or len(coord) <= i :
                x = np.random.randint(residues.shape[2])
                y = np.random.randint(residues.shape[1])
            else :
                x,y = coord[i]
            residues.plot_pixel((x,y), ax = axs[i], color = 'k', label = 'Residues')
            """( one_profile * residues.get_pixels((x,y)).std() ).plot(color = 'b', linestyle = '--',
                label = 'Residues RMS')
            ( -one_profile * residues.get_pixels((x,y)).std() ).plot(color = 'b', linestyle = '--')
            axs[i].plot(channels, noise_map[y, x] * np.ones_like(channels), 'r--',
                label = 'Target RMS')
            axs[i].plot(channels, -noise_map[y, x] * np.ones_like(channels), 'r--')"""
            axs[i].set_title('Pixel ({},{})'.format(x,y))
            axs[i].legend(loc = 'upper right')
        if nProfiles%2 == 1 :
            axs[-1].set_visible(False)

        # Save figure
        fig.tight_layout()
        self._savefig(fig, os.path.join(self.figure_path, 'noise_analysis'), 'residues_profiles.png')


    ### The following methods are usefull if and only if data are artificial
    ### Artificial data (signal, noise and cube) have to be placed in the `data/refs` directory

    def addRefs(self) :
        """ Add artificial noise references (signal, noise and signal+noise cubes) into data directory """
        source = os.path.join('artificial_cubes', self.line)
        destination = os.path.join(self.data_path, 'refs')
        bash = 'cp -r ' + source + ' ' + destination
        subprocess.run(bash, check = True, shell = True,
            stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
        print('AddRefs terminated')

    def plotRefs(self) :
        """ Plot FITS files in `data/refs` directory """
        bash = 'cd ' + self.path + ' && gag21 && cube @../../cube_scripts/plot_fits.cube {} {}'
        self._createDirIfDoesntExist( os.path.join(self.figure_path, 'refs') )
        listFits = self._getFitsInDir( os.path.join(self.data_path, 'refs') )
        for f in listFits :
            try :
                subprocess.run(bash.format(f, 'refs'), check = True, shell = True,
                    stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
            except subprocess.CalledProcessError :
                print('Error raised while plotting refs/{} ignored.'.format(f))
        print('PlotRefs terminated')

    def computeRMSE(self) :
        """ Computes the MSE between the residues and the actual artificial noise """
        noise = fits.open( os.path.join(self.data_path, 'refs', 'noise.fits') )[0].data
        residues = fits.open( os.path.join(self.data_path, 'residues', self.line + '.fits') )[0].data

        # Compute RMSE and normRMSE per pixel
        rmse = np.std( residues-noise, axis = 0 )
        norm_rmse = np.std( residues-noise, axis = 0 ) / np.std( noise, axis = 0 )

        # Plot RMSE map
        fig, ax = plt.subplots(1, 2, figsize = (2 * 6.4, 4.8))

        im = ax[0].imshow(rmse, cmap = 'jet', vmin = 0)
        fig.colorbar(im, ax = ax[0])
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].set_title('RMSE between residues and real noise per pixel')

        im = ax[1].imshow(norm_rmse, cmap = 'jet', vmin = 0)
        fig.colorbar(im, ax = ax[1])
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        ax[1].set_title('RMSE normalized by noise RMS')

        # Save figure
        self._savefig(fig, os.path.join(self.figure_path, 'noise_analysis'), 'RMSE.png')

    def compareProfiles(self, nProfiles : int, coord : list = None, seed : int = 0) :
        """ On `nProfiles` subplots, plot a residues and a real noise profiles """
        refs = fits.open( os.path.join(self.data_path, 'refs', 'noise.fits') )[0].data
        inputs = fits.open( os.path.join(self.data_path, 'inputs', 'noise.fits') )[0].data
        outputs = fits.open( os.path.join(self.data_path, 'outputs', self.line + '.fits') )[0].data

        # Create 2 x n subplots
        rows = nProfiles//2 + nProfiles%2
        fig, axs = plt.subplots(rows, 2, figsize = (2 * 6.4, rows//2 * 4.8))
        axs = axs.flatten()
        np.random.seed(seed)

        # Generate coordinates in None and plot profiles
        channels = list(range(1, inputs.shape[0]+1))
        for i in range(nProfiles) :
            if coord is None or len(coord) <= i :
                x = np.random.randint(inputs.shape[2])
                y = np.random.randint(inputs.shape[1])
            else :
                x,y = coord[i]
            axs[i].plot(channels, inputs[:, y, x], 'k--', label = 'Input')
            axs[i].plot(channels, outputs[:, y, x], 'k', label = 'Output')
            axs[i].plot(channels, refs[:, y, x], 'g', label = 'Signal')
            #axs[i].set_xlabel('Velocity channels')
            #axs[i].set_ylabel('Residues')
            axs[i].set_title('Pixel ({},{})'.format(x,y))
            #axs[i].legend(loc = 'upper left')
        if nProfiles%2 == 1 :
            axs[-1].set_visible(False)

        # Save figure
        fig.tight_layout()
        self._savefig(fig, os.path.join(self.figure_path, 'outputs'), 'comparison_profiles.png')

    def compareNoiseProfiles(self, nProfiles : int, coord : list = None, seed : int = 0) :
        """ On `nProfiles` subplots, plot a residues and a real noise profiles """
        noise = fits.open( os.path.join(self.data_path, 'refs', 'noise.fits') )[0].data
        residues = fits.open( os.path.join(self.data_path, 'residues', self.line + '.fits') )[0].data

        # Create 2 x n subplots
        rows = nProfiles//2 + nProfiles%2
        fig, axs = plt.subplots(rows, 2, figsize = (2 * 6.4, rows//2 * 4.8))
        axs = axs.flatten()
        np.random.seed(seed)

        # Generate coordinates in None and plot profiles
        channels = list(range(1, residues.shape[0]+1))
        for i in range(nProfiles) :
            if coord is None or len(coord) <= i :
                x = np.random.randint(residues.shape[2])
                y = np.random.randint(residues.shape[1])
            else :
                x,y = coord[i]
            axs[i].plot(channels, residues[:, y, x], 'k', label = 'Residues')
            axs[i].plot(channels, noise[:, y, x], 'k--', label = 'Target noise')
            axs[i].plot(channels, residues[:, y, x].std() * np.ones_like(channels), 'b--',
                label = 'Residues RMS')
            axs[i].plot(channels, -residues[:, y, x].std() * np.ones_like(channels), 'b--')
            axs[i].plot(channels, noise[:, y, x].std() * np.ones_like(channels), 'r--',
                label = 'Target RMS')
            axs[i].plot(channels, -noise[:, y, x].std() * np.ones_like(channels), 'r--')
            #axs[i].set_xlabel('Velocity channels')
            #axs[i].set_ylabel('Residues')
            axs[i].set_title('Pixel ({},{})'.format(x,y))
            #axs[i].legend(loc = 'upper left')
        if nProfiles%2 == 1 :
            axs[-1].set_visible(False)

        # Save figure
        fig.tight_layout()
        self._savefig(fig, os.path.join(self.figure_path, 'noise_analysis'), 'comparison_noise_profiles.png')

    ### The following methods must be called ONLY after plotting

    def clean_temporary(self) :
        """ Clean temporary files and directories generated during the plots """
        temp_paths = glob.glob( os.path.join(self.path, '**', 'tmp'), recursive = True )
        for path in temp_paths :
            shutil.rmtree(path)
        print('Temporary files cleaned')