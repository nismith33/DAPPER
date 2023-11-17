#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 11:35:53 2023

This script processes  objective analyses - Gouretski and Reseghetti (2010) 
taken from
https://hadleyserver.metoffice.gov.uk/en4/download-en4-2-2.html

@author: ivo

"""
from datetime import datetime, timedelta
import numpy as np
import netCDF4 as nc
import xarray as xr
import re
import os
from matplotlib import pyplot as plt
import sklearn as sl
import shapely
import matplotlib as mpl
import pickle as pkl

# Earth radius
EARTH_RADIUS = 6.3781e6  # m
#SECONDS IN YEAR
YEAR = 365.25 * 86400 #s

def kelvin2celsius(temp):
    """
    Convert temperatures in Kelvin to Celsius. 

    Parameters
    ----------
    temp : float | numpy array of float
        Array with temperature in Kelvin

    Returns
    -------
    Array with temperature in Celsius

    """
    return temp - 273.14


class HadleyObs:
    """
    Class coordinating reading MetOffice Hadley server files. 

    Attributes
    ----------
    topdir : str 
        Directory containing Hadley files and subdirectories 
        with Hadley files.


    """

    def __init__(self, topdir, chunck_size=None):
        """ Class constructor. """
        self.chunck_size = chunck_size

        # Directory containing
        self.topdir = topdir

        # Search for suitable files.
        self._pattern = re.compile(
            "EN.4.2.2.f.analysis.g10.([0-9]{4,4})([0-9]{2,2}).nc")

        # Search for all files that match pattern in topdir and its subdirectories.
        self.list_files()

    def list_files(self):
        """ Create a list of path to all Hadley files and their times. """
        self.filepaths, self.times = [], []

        # Obtain all files that have a valid file format.
        for root, dirs, files in os.walk(self.topdir):
            for file in files:
                time = self.extract_time(file)

                # If time is returned, filename matches pattern.
                if time is not None:
                    self.times += [time]
                    self.filepaths += [os.path.join(root, file)]

        # Turn output in numpy arrays.
        self.times = np.array(self.times)
        self.filepaths = np.array(self.filepaths)

        # Sort files by ascending time.
        isort = np.argsort(self.times)
        self.times, self.filepaths = self.times[isort], self.filepaths[isort]

    def extract_time(self, filename):
        """ 
        Extract time from filename 

        Parameters
        ----------
        filename : str
            Filename to be stripped for time. 

        Returns 
        -------
        Time as datetime object or None if not a valid filename. 

        """
        # Try match filename to pattern.
        match = re.match(self._pattern, filename)

        # Return time if possible.
        if match is None:
            return None
        else:
            return datetime(int(match[1]), int(match[2]), 1)

    def read(self, time_bounds=(datetime(1800, 1, 1), datetime(2100, 1, 1))):
        """ 
        Read all files within time range into an Xarray object.

        Parameters
        ----------
        time_bounds : tuple of datetime object
            Time range to be read from files. 


        """
        # Select files within specified time range.
        
        mask = np.logical_and(self.times >= time_bounds[0],
                              self.times <= time_bounds[1])

        # Combine filedata into single Xarray object.
        if self.chunck_size:
            datas = xr.open_mfdataset(self.filepaths[mask], 
                                      chunks={'time':self.chunck_size},
                                      engine='netcdf4', combine='nested',
                                      concat_dim='time')
            output = datas
        else:
            datas = []
            for file in self.filepaths:
                datas.append(xr.open_dataset(file))
            output = xr.concat(datas, 'time')
            
        # Add longitude as number in range [-180,180)
        output = output.assign(lon180=np.mod(output.lon+180, 360)-180)

        # Add volumes
        output = self.add_area(output)
        output = self.add_volume(output)

        # Return Xarray object.
        return output

    def add_area(self, data):
        """ 
        Calculate grid cell area.        
        """
        # Longitude and latitude grids.
        lon, lat = np.meshgrid(data['lon'], data['lat'])
        lon180, lat = np.meshgrid(data['lon180'], data['lat'])

        # Jacobian
        Jx = EARTH_RADIUS * np.cos(np.deg2rad(lat)) * np.deg2rad(1)
        Jy = EARTH_RADIUS * np.deg2rad(1) * np.ones_like(lat)

        # Calculate areas
        dx = np.ones_like(lon) * np.nan
        dx180 = np.ones_like(lon) * np.nan
        dy = np.ones_like(lon) * np.nan
        
        dx[1:-1,1:-1] = 0.5*(lon[1:-1,2:] - lon[1:-1,:-2]) * Jx[1:-1,1:-1]
        dx180[1:-1,1:-1] = 0.5*(lon180[1:-1,2:] - lon180[1:-1,:-2]) * Jx[1:-1,1:-1]
        dx = np.minimum(np.abs(dx), np.abs(dx180))
        dy[1:-1,1:-1] = 0.5*(lat[2:, 1:-1] - lat[:-2, 1:-1]) * Jy[1:-1,1:-1]
        areas = dx*dy
        
        data = data.assign(area=(['lat', 'lon'], np.abs(areas)))
        data = data.assign(dx=(['lat','lon'], np.abs(dx)))
        data = data.assign(dy=(['lat','lon'], np.abs(dy)))        
        data = data.assign(width=(['lat'], np.nansum(np.abs(dy),axis=1)))
        data = data.assign(height=(['lon'], np.nansum(np.abs(dx),axis=0)))

        return data

    def add_volume(self, data):
        """ 
        Calculate grid cell volume. 
        """
        area = np.array(data['area'])[None, :, :]
        depth = np.array(data['depth']).reshape((-1, 1, 1))

        # Calculate depths.
        dz = np.ones((np.size(depth, 0), np.size(
            area, 1), np.size(area, 2))) * np.nan
        dz[1:-1, :, :] = 0.5*(depth[2:, :, :] - depth[:-2, :, :]) * area
        
        #Expand to top 
        dz[0,:,:] = dz[1,:,:]
        

        return data.assign(volume=(['depth', 'lat', 'lon'], np.abs(dz)))


class StommelClusterer:
    """ 
    Divide the north Atlantic in a poleward and equator part and divide 
    each in a surface layer and deep ocean layer. 

    Parameters
    ----------
    data : xarray object 
        Object with Hadley observations on grid. 

    """

    def __init__(self, data):
        """ Class constructor. """
        self.data = data

    def select_atlantic(self):
        """ Select grid points in North-Atlantic. """

        # Polygon around the north-atlantic (lat, lon)
        points = [(35.955065, -5.606114), (0., 0.095069),
                  (0., -60.626643), (10.671971, -60.718661),
                  (17.958394, -60.683383), (27.043391, -79.990489),
                  (34.899044, -87.998876), (60., -64.),
                  (70.165621, -45.0),
                  (70., 23.5), (54.168530, -9.246100)]
        
        points = [(35.955065, -5.606114), (23.5, 0.095069),
                  (23.5, -60.626643), (23.5, -60.718661),
                  (23.5, -60.683383), (27.043391, -79.990489),
                  (34.899044, -87.998876), (60., -64.),
                  (70.165621, -45.0),
                  (70., 23.5), (54.168530, -9.246100)]
        
        poly = shapely.Polygon(points)

        # Mask with grid points in North Atlantic.
        lon, lat = np.meshgrid(self.data.lon180, self.data.lat)
        ix, iy = np.meshgrid(range(self.data.lon180.shape[0]),
                             range(self.data.lat.shape[0]))
        mask = ~np.isnan(self.data['temperature'][0, 0])

        # Select indices of points in mask.
        indices = []
        for ix1, iy1 in zip(ix.ravel(), iy.ravel()):
            point = shapely.Point(lat[iy1, ix1], lon[iy1, ix1])
            if mask[iy1, ix1] and poly.contains(point):
                indices += [(iy1, ix1)]

        # Return indices.
        return indices

    def _simple_circle(self, x, y):
        """
        Calculate distance between between points on earth. 

        Instead of great circle distance a simple Jacobian is used. 

        Parameters
        ----------
        x : tuple of floats 
            One point on earth. 
        y : tuple of floats
            Other point on earth.

        Returns
        -------
        Distance between points. 

        """
        # Difference lattitude on longitude.
        dlat = (x[1] - y[1])
        dlon = (x[0] - y[0])
        # Halfway lattitude
        mlat = 0.5*np.cos(np.deg2rad(x[1])) + 0.5*np.cos(np.deg2rad(y[1]))
        # Return distance.
        return dlat**2 + dlon**2 * mlat**2

    def cluster_horizontal(self, indices, min_depth=2e3):
        """ Divide profiles into pole/equator classes. 

        Parameters
        ----------
        indices : list of 2D int tuples
            Horizontal grid indices of that need to be divided into two classes.
        min_depth: float 
            Profiles need be as least as deep as this to be included.

        Returns
        -------
        List with each element a list of 3D indices in that cluster. 

        """
        from sklearn.pipeline import Pipeline
        from sklearn.impute import KNNImputer, SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering

        # Variables
        temp = np.array(self.data['temperature'].mean('time'))
        salt = np.array(self.data['salinity'].mean('time'))
        lat = np.array(self.data['lat'])
        lon = np.array(self.data['lon180'])

        # Select only those profiles deeper 2km.
        km2 = int(np.sum(self.data['depth'] <= min_depth))
        mask = np.sum(~np.isnan(temp), axis=0) >= km2
        indices = [(iy1, ix1) for (iy1, ix1) in indices if mask[iy1, ix1]]

        # Create features from latitude, temperature and salinity
        features = []
        for (iy1, ix1) in indices:
            features += [np.concatenate((lat[iy1]*np.ones_like(temp[:km2, iy1, ix1]),
                                         temp[:km2, iy1, ix1],
                                         salt[:km2, iy1, ix1],))]
        features = np.array(features)

        # Rescale features and cluster using K-means
        pipeline = Pipeline([('norm', StandardScaler()),
                             ('clustering', KMeans(n_clusters=2, n_init='auto'))])
        predictions = pipeline.fit_predict(features)

        # Group indices by cluster
        clusters = [np.compress(predictions == cluster, indices, axis=0).reshape((-1, 2)) for
                    cluster in np.unique(predictions)]

        # Return clusters.
        return clusters

    def cluster_vertical(self, indices):
        """
        Divide volume specified by indices in upper and lower part. 

        Parameters
        ----------
        indices : list of 3D int tuples
            Grid indices of volume that need to split in upper/lower part.

        Returns
        -------
        List with each element a list of 3D indices in that cluster. 

        """

        from sklearn.pipeline import Pipeline
        from sklearn.impute import KNNImputer, SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering

        # Depths
        z = np.array(self.data['depth'])
        iz = range(len(z))

        # Observed fields
        temp = np.array(self.data['temperature'].mean('time'))
        salt = np.array(self.data['salinity'].mean('time'))
        mask = ~np.isnan(temp)

        # Add depth index
        indices = [(iz1, iy1, ix1)
                   for (iy1, ix1) in indices for iz1 in iz if mask[iz1, iy1, ix1]]

        # Create features
        features = []
        for (iz1, iy1, ix1) in indices:
            features += [np.array([z[iz1], z[iz1],
                                   temp[iz1, iy1, ix1], salt[iz1, iy1, ix1]])]
        features = np.array(features)

        # Cluster
        pipeline = Pipeline([('norm', StandardScaler()),
                             ('clustering', KMeans(n_clusters=2, n_init='auto'))])
        predictions = pipeline.fit_predict(features)

        # Group indices by cluster
        clusters = [np.compress(predictions == cluster, indices, axis=0).reshape((-1, 3)) for
                    cluster in np.unique(predictions)]

        return clusters

    def cluster1d_vertical(self, indices):
        """
        Divide volume specified by indices in upper and lower part. 

        Parameters
        ----------
        indices : list of 3D int tuples
            Grid indices of volume that need to split in upper/lower part.

        Returns
        -------
        List with each element a list of 3D indices in that cluster. 

        """
        from sklearn.pipeline import Pipeline
        from sklearn.impute import KNNImputer, SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering

        def unique_order(iterable):
            """ Return unique values preserving order. """
            return list(dict.fromkeys(iterable).keys())

        # Depths
        z = np.array(self.data['depth'])
        iz = np.arange(len(z))

        # Observed fields
        temp = np.array(self.data['temperature'].mean('time'))
        salt = np.array(self.data['salinity'].mean('time'))

        # Cluster
        pipeline = Pipeline([('norm', StandardScaler()),
                             ('clustering', KMeans(n_clusters=2, n_init='auto'))])

        clusters = [[], []]
        for (iy1, ix1) in indices:
            # Depths for which observations are available.
            mask = ~np.isnan(temp[:, iy1, ix1])
            # Cluster profile.
            features = np.array([z[mask], z[mask],
                                 temp[mask, iy1, ix1], salt[mask, iy1, ix1]]).T
            predictions = pipeline.fit_predict(features)

            # Combine predictions with indices
            predictions = np.array([predictions, iz[mask],
                                    iy1*np.ones_like(iz[mask]),
                                    ix1*np.ones_like(iz[mask])])

            for cluster, prediction in enumerate(unique_order(predictions[0])):
                mask = predictions[0] == prediction
                clusters[cluster] += list(predictions[1:, mask].T)

        # Return lists of indices for each cluster.
        return clusters
    
    def cluster_top(self, indices):
        """ Seperate top layer from deeper layers. """
        clusters = [[], []]
        
        # Observed fields
        temp = np.array(self.data['temperature'].mean('time'))
        
        for (iy1, ix1) in indices:
            # Depths for which observations are available.
            mask = ~np.isnan(temp[:, iy1, ix1])
            
            if not mask[0]:
                continue 
            
            for iz1, mask1 in enumerate(mask):
                z1 = self.data['depth'][iz1]
                if mask1 and iz1<=0:
                    clusters[0] += [(iz1,iy1,ix1)]
                elif mask1 and z1<np.inf:
                    clusters[1] += [(iz1,iy1,ix1)]

        # Return lists of indices for each cluster.
        return clusters
   
#%% Plotting routines     

def plot_cluster3d(ax, data, indices):
    """
    Plot position of selected grid points with color indicating their cluster.

    Parameters
    ----------
    ax : matplotlib.axis object 
        Axes in which plot will drawn. 
    indices : list of lists 
        List with each element of 3D indices in that cluster. 

    """

    lon, lat, depth, no = [], [], [], []
    for n, cluster in enumerate(indices):
        lon += [data['lon180'][ix] for (iz, iy, ix) in cluster]
        lat += [data['lat'][iy] for (iz, iy, ix) in cluster]
        depth += [-data['depth'][iz]*1.0e-3 for (iz, iy, ix) in cluster]
        no += [n for _ in cluster]

    m = ax.scatter3D(lon, lat, depth, s=1., c=no, vmin=0, vmax=len(indices),
                     cmap=mpl.colormaps.get_cmap('tab10'))

    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_zlabel('z [km]')

    return ax


def plot_cluster2d(ax, data, indices):
    """
    Plot position of selected grid points with color indicating their cluster.

    Parameters
    ----------
    ax : matplotlib.axis object 
        Axes in which plot will drawn. 
    indices : list of lists 
        List with each element of 3D indices in that cluster. 

    """

    lon, lat, no = [], [], []
    for n, cluster in enumerate(indices):
        lon += [data['lon180'][ix] for (iy, ix) in cluster]
        lat += [data['lat'][iy] for (iy, ix) in cluster]
        no += [n for _ in cluster]

    m = ax.scatter(lon, lat, s=2., c=no, vmin=0, vmax=len(indices),
                   cmap=mpl.colormaps.get_cmap('tab10'))

    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')

    return ax


#%% Processing routines

def average_cluster(data, indices):
    """ 
    Inverse variance weighted averaging over dimension ind

    Parameters
    ----------
    data : xarray object 
        Object containing observed data arrays. 
    field : str 
        Field to be averaged. 

    """
    output = {}
    indices = np.array(indices)
    indices2d = np.unique(indices[:, 1:], axis=0).T
    indices3d = indices.T

    # Area indices.
    data2d = data.isel(lat=xr.DataArray(indices2d[0], dims='ind2d'),
                       lon=xr.DataArray(indices2d[1], dims='ind2d'),
                       )
    
    # Average lon, lat
    for field in ['lat', 'lon']:
        data2d[field] = (data2d[field] * data2d['area']
                         ).mean('ind2d') / data2d['area'].mean('ind2d')
        output = {**output, field: data2d[field]}

    # Total area.
    output = {**output, 'area': data2d['area'].sum('ind2d')}

    # Volume averages
    data3d = data.isel(depth=xr.DataArray(indices3d[0], dims='ind3d'),
                       lat=xr.DataArray(indices3d[1], dims='ind3d'),
                       lon=xr.DataArray(indices3d[2], dims='ind3d'),
                       )
    
    # Observed fields.
    for field in ['temperature', 'salinity']:
        w = data3d['volume'] / data3d[field+'_uncertainty']**2
        data3d[field] = (data3d[field] * w).mean('ind3d') / w.mean('ind3d')
        data3d[field +'_uncertainty'] = (data3d['volume'].mean('ind3d') / w.mean('ind3d'))**0.5
        output = {**output, field: data3d[field], field +
                  '_uncertainty': data3d[field+'_uncertainty']}

    # Depth
    data3d['depth'] = (data3d['depth']*data3d['volume']
                       ).mean('ind3d') / data3d['volume'].mean('ind3d')
    output = {**output, 'depth': data3d['depth']}

    # Volume
    output = {**output, 'volume': data3d['volume'].sum('ind3d')}
      

    return xr.Dataset(output)

def cross_section(data, indices, label):
    """
    Calculate the cross-section between the boxes. 
    """
    indices = np.array(indices)
    indices3d = indices.T
    
    if label == 'pole':
        interface_index = np.min 
    elif label == 'equator':
        interface_index = np.max
    
    # Interface 
    mask = np.zeros_like(indices3d[0])
    for ilon in np.unique(indices3d[2]):
        ilat = interface_index(indices3d[1][indices3d[2]==ilon])
        mask = np.logical_or(mask, np.logical_and(indices3d[1]==ilat,indices3d[2]==ilon))
    
    data3d = data.isel(depth=xr.DataArray(indices3d[0][mask], dims='ind3d'),
                       lat=xr.DataArray(indices3d[1][mask], dims='ind3d'),
                       lon=xr.DataArray(indices3d[2][mask], dims='ind3d'),
                       )
    
    areaInter = data3d['volume'].sum('ind3d') / data3d['dy'].mean('ind3d')
    return float(areaInter)


def label_indices(data, indices):
    """
    Given a set of indices sets determine for each set whether they corresponds
    to pole or equator box and surface or ocean box. 
    
    """
    if len(indices) != 4:
        msg = "4 clusters must be provided."
        raise ValueError(msg)

    # Spatial average.
    outputs = []
    for indices_cluster in indices:
        outputs.append(average_cluster(data, indices_cluster))

    # Sort clusters first latitude than by depth
    lats = [float(output['lat']) for output in outputs]
    depths = [float(output['depth']) for output in outputs]
    isort = np.lexsort((depths, lats))

    labels = [('equator','surface'),('equator','ocean'),
              ('pole','surface'),('pole','ocean')]
    labels = np.take(labels, isort, axis=0)
    
    return labels
    

def create_yy(data, indices, dt=1):
    """ 
    This function creates a file containing the numpy arrays for yy and
    error variance. 

    Parameters
    ----------
    data : xarray object 
        Object containing on Hadley observations.
    indices : list of xarray objects. 
        Object containing time series of volume-averaged time series produced
        by spatial_average for each cluster.
    dt : int > 0 
        Averaging period for observations.

    """
    dt = int(dt)
    
    #Label different data
    labels = label_indices(data,indices)
    
    # Values of observations 
    for label, index in zip(labels, indices):
        if all(label==['pole','ocean']):
            pole = average_cluster(data, index)
        if all(label==['equator','ocean']):
            equator = average_cluster(data, index)

    # Observations
    yy = np.array([pole['temperature'][1:], equator['temperature'][1:], 
                   pole['salinity'][1:], equator['salinity'][1:]])
    yy[:2] = kelvin2celsius(yy[:2])
    yy = list(yy.T)
    
    #Average in time
    yy=yy[:int(np.size(yy,0)/dt)*dt]
    yy=np.reshape(yy,(-1,dt,np.size(yy,1)))
    yy=np.mean(yy, axis=1)

    # Observational error covariances
    R = np.array([pole['temperature_uncertainty'].max(),
                  equator['temperature_uncertainty'].max(),
                  pole['salinity_uncertainty'].max(),
                  equator['salinity_uncertainty'].max()])
    R = R**2

    # Observationa error bias
    mu = np.zeros_like(R)

    return yy, mu, R   
    
def create_surface(data, indices, period=np.timedelta64(int(YEAR),'s')):
    """ 
    This function calculates the mean temperature and salinity over the 
    whole time period as well as std. deviation in temperature and salinity 
    over the period. 

    Parameters
    ----------
    data : xarray object 
        Object containing on Hadley observations.
    indices : list of xarray objects. 
        Object containing time series of volume-averaged time series produced
        by spatial_average for each cluster.

    """
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression(fit_intercept=False, copy_X=False)
    
    # Values of observations 
    labels = label_indices(data,indices)
    for label, index in zip(labels, indices):
        if all(label==['pole','surface']):
            pole = average_cluster(data, index)
        if all(label==['equator','surface']):
            equator = average_cluster(data, index)
            
    #Calculate mean surface temp/salt and standard deviation. 
    harmonics = {'harmonic_angular' : np.array([2 * np.pi / float( period / np.timedelta64(1,'s'))])}
    for field in ['temperature', 'salinity']:
        key = 'surface_'+field+'_harmonic'
        harmonics[key] = []
        for box, data in enumerate([pole, equator]):
            #Time relative to first time. 
            times = np.array(data['time'] - data['time'][0]) / np.timedelta64(1,'s')
            #Weights data 
            w = 1/data[field+'_uncertainty']**2
            #Base function for regression. 
            X      = np.ones((len(w),
                              1+2*len(harmonics['harmonic_angular'])))
            for n,omega in enumerate(harmonics['harmonic_angular']):
                X[:,1+2*n] = np.cos(omega*times)
                X[:,2+2*n] = np.sin(omega*times)
            #Fit 
            if field=='temperature':
                regressor.fit(X, kelvin2celsius(data[field]), sample_weight=w)
            else:
                regressor.fit(X, data[field], sample_weight=w)
            #Add to data 
            harmonics[key].append(np.array(regressor.coef_))       
            
        #Create output
        harmonics[key] = np.array(harmonics[key]).T
        
    return harmonics

#%% Run commands


# Directory containing files downloaded from Hadley server.
DIR = "/home/oceancirculation059/dpr_data/stommel"
# Read in data from files.
obs = HadleyObs(DIR, chunck_size='512MB')
output = obs.read()
# Start clustering.
clusterer = StommelClusterer(output)
# Indices of points in North Atlantic and
# divide in pole and equatorial box.
h_indices = clusterer.cluster_horizontal(clusterer.select_atlantic())
# Divide each horizontal cluster in 2 vertical clusters.
hv_indices = []
for indices in h_indices:
    hv_indices += clusterer.cluster_top(indices)

# Create time series for each of the clusters and plot it.
plt.close('all')
fig = plt.figure(figsize=(11, 8))
axes = fig.subplots(2, 2)
avg_outputs = []
for n, indices in enumerate(hv_indices):
    avg_outputs.append(average_cluster(output, indices))

    label = 'depth={:.0f}, lat={:.1f}'.format(float(avg_outputs[-1]['depth']),
                                              float(avg_outputs[-1]['lat']))
    for field, ax in zip(['temperature', 'salinity', 'temperature_uncertainty', 'salinity_uncertainty'],
                         axes.ravel()):
        avg_outputs[-1][field].plot.line(x='time', ax=ax, label=label)

    plt.legend(loc='upper right')

for ax in axes.ravel():
    ax.grid()

# Create observations that can be assimilated.
yy, mu, R = create_yy(output, hv_indices)

# Calculate surface forcing
surface_data = create_surface(output, hv_indices)

#Save all relevant data into file. 
# Save into file
boxed_hadley = {'yy': yy, 'R': R, 'mu': mu, **surface_data}
labels = label_indices(output, hv_indices)
geo = {'cross':0.,'area':0.,'vol':0.}
for label, index in zip(labels, hv_indices):
    avg = average_cluster(output, index)
    if all(label==['pole','ocean']):
        geo['cross'] = geo['cross']+0.5*cross_section(output,index,label[0])
        geo['area']  = geo['area']+float(avg['area'])
        geo['vol']   = geo['vol']+float(avg['volume'])
    if all(label==['equator','ocean']):
        geo['cross'] = geo['cross']+0.5*cross_section(output,index,label[0])
        geo['area']  = geo['area']+float(avg['area'])
        geo['vol']   = geo['vol']+float(avg['volume'])
        
dz = geo['vol'] / geo['area'] 
dx = geo['cross'] / dz
for label, index in zip(labels, hv_indices):
    avg = average_cluster(output, index)
    #Depth must be equal for both. 
    dy = float(avg['volume']) / (dx*dz)
    geo = {'dz':dz,'dx':dx,'dy':dy}
    if all(label==['pole','ocean']):
        boxed_hadley = {**boxed_hadley, 'geo_pole':geo}
    if all(label==['equator','ocean']):
        boxed_hadley = {**boxed_hadley, 'geo_eq':geo}
        
with open(os.path.join(DIR, 'boxed_hadley.pkl'), 'wb') as stream:
    pkl.dump(boxed_hadley, stream)

# Plot clusters.
fig = plt.figure(figsize=(11, 8))
axes = fig.subplots(1, 2, subplot_kw={'projection': '3d'})
# Plot cluster for each point in horizontal.
axes[0] = plot_cluster2d(axes[0], output, h_indices)
# Plot cluster for each point in horizontal and vertical.
axes[1] = plot_cluster3d(axes[1], output, hv_indices)
