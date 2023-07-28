#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 11:35:53 2023

This script processes objective analysis.

Analysis are taken from 
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

DIR = "/home/ivo/Downloads/EN.4.2.2.analyses.g10.2019"
EARTH_RADIUS = 6.3781e6  # m

atlantic = {'lat': (0, 65), 'lon': (-76, -6), 'depth': (0, 1e4)}


def sdeg(deg):
    return np.mod(deg+180., 360.)-180.


def ideg(deg):
    return np.mod(deg, 360.)


class HadleyObs:
    """
    Class coordinating reading files. 

    """

    def __init__(self, topdir):
        """ Class constructor. """
        self.topdir = topdir

        # Search for suitable files.
        self.pattern = re.compile(
            "EN.4.2.2.f.analysis.g10.([0-9]{4,4})([0-9]{2,2}).nc")
        self.list_files()

    def list_files(self):
        """ Create a list of path to all Hadley files and their times. """
        self.filepaths, self.times = [], []

        # Obtain all files that have a valid file format.
        for root, dirs, files in os.walk(self.topdir):
            for file in files:
                time = self.extract_time(file)

                if time is not None:
                    self.times += [time]
                    self.filepaths += [os.path.join(root, file)]

        # Turn into numpy arrays.
        self.times = np.array(self.times)
        self.filepaths = np.array(self.filepaths)

        # Sort by ascending time.
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
        match = re.match(self.pattern, filename)

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

        datas = []
        mask = np.logical_and(self.times >= time_bounds[0],
                              self.times <= time_bounds[1])
        for file in self.filepaths[mask]:
            datas.append(xr.open_dataset(file))

        # Return arrays
        output = xr.concat(datas, 'time')
        output = output.assign(lon180=np.mod(output.lon+180, 360)-180)
        return output

    def calculate_volume(self, data):
        R = 2*np.pi/360 * EARTH_RADIUS  # m
        depth = np.array(data['depth'])
        lon = np.array(data['lon180'])
        lat = np.array(data['lat'])

        vol = np.reshape(.5*(depth[2:] - depth[:-2]), (-1, 1, 1))
        vol = vol * np.reshape(.5*(lat[2:] - lat[:-2]), (1, -1, 1)) * R
        vol = (np.reshape(.5*(lon[2:] - lon[:-2]), (1, 1, -1)) *
               np.reshape(np.cos(np.deg2rad(lat[1:-1])), (1, -1, 1))) * R * vol

        return vol


def select(data, bounds):
    lon = data.lon
    lat = data.lat
    depth = data.depth

    if 'lon' in bounds:
        mask = np.logical_and(data.lon180 >= bounds['lon'][0],
                              data.lon180 <= bounds['lon'][1])
        lon = lon[mask]
    if 'lat' in bounds:
        mask = np.logical_and(lat >= bounds['lat'][0], lat <= bounds['lat'][1])
        lat = lat[mask]
    if 'depth' in bounds:
        mask = np.logical_and(
            depth >= bounds['depth'][0], depth <= bounds['depth'][1])
        depth = depth[mask]

    return output.sel(lon=lon, lat=lat, depth=depth)


obs = HadleyObs(DIR)
output = obs.read()
volume = obs.calculate_volume(output)
sel = select(output, atlantic)


class cluster_basins:

    def __init__(self, data):
        self.data = data

    def select_atlantic(self):
        """ Select grid points in North-Atlantic. """
        
        # Polygon around north-atlantic (lat, lon)
        points = [(35.955065, -5.606114), (0., 0.095069),
                  (0., -60.626643), (10.671971, -60.718661),
                  (17.958394, -60.683383), (27.043391, -79.990489),
                  (34.899044, -87.998876), (60.,-64.),
                  (70.165621, -45.0), 
                  (70., 23.5), (54.168530, -9.246100)]
        poly = shapely.Polygon(points)

        # Grid
        lon, lat = np.meshgrid(self.data.lon180, self.data.lat)
        ix, iy = np.meshgrid(range(self.data.lon180.shape[0]),
                             range(self.data.lat.shape[0]))
        mask = ~np.isnan(self.data['temperature'][0,0])
        
        #Points in basin.
        indices = []
        for ix1, iy1 in zip(ix.ravel(), iy.ravel()):
            point = shapely.Point(lat[iy1,ix1], lon[iy1,ix1])
            if mask[iy1,ix1] and poly.contains(point):
                indices += [(iy1,ix1)]
                
        return indices
    
    def _simple_circle(self, x, y):
        dlat = (x[1] - y[1])
        dlon = (x[0] - y[0])
        mlat = .5*np.cos(np.deg2rad(x[1]))+.5*np.cos(np.deg2rad(y[1]))
        return dlat**2 + dlon**2 * mlat**2
    
    def cluster_horizontal(self, indices):
        """ Divide profiles into pole/equator classes. """
        from sklearn.pipeline import Pipeline
        from sklearn.impute import KNNImputer, SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
        
        #Variables
        temp = np.array(self.data['temperature'].mean('time'))
        salt = np.array(self.data['salinity'].mean('time'))
        lat = np.array(self.data['lat'])
        lon = np.array(self.data['lon180'])
        
        #Select only those profiles deeper 2km. 
        km2 = int(np.sum(self.data['depth'] <= 2e3))
        mask = np.sum(~np.isnan(temp), axis=0) >= km2
        indices = [(iy1,ix1) for (iy1,ix1) in indices if mask[iy1,ix1]]
        
        #Create features from latitude, 2km temperature and salinity
        features = []
        for (iy1,ix1) in indices:
            features += [np.concatenate((lat[iy1:iy1+1],
                                         temp[:km2,iy1,ix1],
                                         salt[:km2,iy1,ix1],))]
        features = np.array(features)
        
        #Rescale features
        pipeline = Pipeline([('norm', StandardScaler()),
                             ('clustering', KMeans(n_clusters=2))])
        
        predictions = pipeline.fit_predict(features)
        
        #Group indices by cluster
        clusters = [np.compress(predictions==cluster, indices, axis=0).reshape((-1,2)) for 
                    cluster in np.unique(predictions)]
            
        return clusters
    
    def cluster_vertical(self, indices):
        from sklearn.pipeline import Pipeline
        from sklearn.impute import KNNImputer, SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
        
        #depths
        z = np.array(self.data['depth'])
        iz = range(len(z))
        
        #fields
        temp = np.array(self.data['temperature'].mean('time'))
        salt = np.array(self.data['salinity'].mean('time'))
        mask = ~np.isnan(temp)
        
        #Add depth index
        indices = [(iz1,iy1,ix1) for (iy1,ix1) in indices for iz1 in iz if mask[iz1,iy1,ix1]]
        
        #Create features
        features = []
        for (iz1,iy1,ix1) in indices:
            features += [np.array([z[iz1],temp[iz1,iy1,ix1],salt[iz1,iy1,ix1]])]
        features = np.array(features)
        
        #Cluster
        pipeline = Pipeline([('norm', StandardScaler()),
                             ('clustering',KMeans(n_clusters=2))])
        predictions = pipeline.fit_predict(features)
            
        
        #Group indices by cluster
        clusters = [np.compress(predictions==cluster, indices, axis=0).reshape((-1,3)) for 
                    cluster in np.unique(predictions)]
        
        return clusters
    
    def cluster1d_vertical(self, indices):
        from sklearn.pipeline import Pipeline
        from sklearn.impute import KNNImputer, SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
        
        #depths
        z = np.array(self.data['depth'])
        
        #fields
        temp = np.array(self.data['temperature'].mean('time'))
        salt = np.array(self.data['salinity'].mean('time'))
        
        #Cluster
        pipeline = Pipeline([('norm', StandardScaler()),
                             ('clustering',KMeans(n_clusters=2))])
        
        clusters = [[], []]
        for (iy1,ix1) in indices:
            mask = ~np.isnan(temp[:,iy1,ix1])
            features = np.array([z[mask], temp[mask,iy1,ix1], salt[mask,iy1,ix1]]).T
            predictions = pipeline.fit_predict(features)
            
            for iz1, cluster in enumerate(predictions):
                clusters[cluster] += [(iz1,iy1,ix1)]
        
        return clusters
        
    
    def scatter_plotH(self, indices, field, iz):
        plt.figure()
        values = np.array(self.data[field].mean('time'))
        lon = np.array([self.data['lon180'][ix1] for (iy1,ix1) in indices])
        lat = np.array([self.data['lat'][iy1] for (iy1,ix1) in indices])
        values = np.array([values[iz,iy1,ix1] for (iy1, ix1) in indices])
        
        plt.scatter(lon,lat,c=values)
        plt.colorbar()
        
    def plotH_clusters(self, clusters): 
        plt.figure()
        for no, indices in enumerate(clusters):
            lon = np.array([self.data['lon180'][ix1] for (iy1,ix1) in indices])
            lat = np.array([self.data['lat'][iy1] for (iy1,ix1) in indices])
            plt.scatter(lon,lat,c=np.ones_like(lon)*no,vmin=0,vmax=len(clusters)-1)
            
        plt.colorbar()
            
    def scatter_plotV(self, indices, field):
        plt.figure()
        values = np.array(self.data[field].mean('time'))
        depth = np.array([self.data['depth'][iz1] for (iz1,iy1,ix1) in indices])
        lat = np.array([self.data['lat'][iy1] for (iz1,iy1,ix1) in indices])
        values = np.array([values[iz1,iy1,ix1] for (iz1,iy1, ix1) in indices])
        
        plt.scatter(lat,-depth,c=values)
        plt.colorbar()
        
    def plotV_clusters(self, clusters): 
        plt.figure()
        for no, indices in enumerate(clusters):
            depth = np.array([self.data['depth'][iz1] for (iz1,iy1,ix1) in indices])
            lat = np.array([self.data['lat'][iy1] for (iz1,iy1,ix1) in indices])
            plt.scatter(lat,-depth,c=np.ones_like(lat)*no,vmin=0,vmax=len(clusters)-1)
            
        plt.colorbar()
        
        



C = cluster_basins(output)
indices = C.select_atlantic()
clusters = C.cluster_horizontal(indices)

plt.close('all')
C.plotH_clusters(clusters)

clusters0 = C.cluster_vertical(clusters[0])
clusters1 = C.cluster_vertical(clusters[1])
C.plotV_clusters(clusters0)
C.scatter_plotV(np.concatenate((clusters0[1],)), 'salinity')
C.scatter_plotV(np.concatenate((clusters1[1],)), 'salinity')

#cluster = cluster_basins(sel)
#feat = cluster.build_features()
#feat_norm = cluster.preprocess_horz(feat)
#cluster_no = cluster.cluster_horz(feat_norm)
