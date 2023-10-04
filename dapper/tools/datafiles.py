"""Manage the data files created by DAPPER."""

import os
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path

import dill
from netCDF4 import Dataset
from tqdm.auto import tqdm

import dapper.tools.remote.uplink as uplink
from dapper.dpr_config import rc
from dill.tests.test_classdef import nc

XP_TIMESTAMP_TEMPLATE = "run_%Y-%m-%d__%H-%M-%S"


def create_run_dir(save_as, mp):
    """Validate `save_as` and create dir `rc.dirs.data / save_as` and sub-dirs.

    The data gets saved here unless `save_as` is `False`/`None`.

    Note: multiprocessing (locally or in the cloud) requires saving/loading data.
    """
    if save_as in [None, False]:
        assert not mp, "Multiprocessing requires saving data."
        # Parallelization w/o storing is possible, especially w/ threads.
        # But it involves more complicated communication set-up.
        def xpi_dir(*args): return None

    else:
        save_as = rc.dirs.data / Path(save_as).stem
        save_as /= datetime.now().strftime(XP_TIMESTAMP_TEMPLATE)
        os.makedirs(save_as)
        print(f"Experiment stored at {save_as}")

        def xpi_dir(i):
            path = save_as / str(i)
            os.mkdir(path)
            return path

    return save_as, xpi_dir


def find_latest_run(root: Path):
    """Find the latest experiment (dir containing many)"""
    def parse(d):
        try:
            return datetime.strptime(d.name, XP_TIMESTAMP_TEMPLATE)
        except ValueError:
            return None
    dd = [e for e in (parse(d) for d in root.iterdir()) if e is not None]
    d = max(dd)
    d = datetime.strftime(d, XP_TIMESTAMP_TEMPLATE)
    return d


def load_HMM(save_as):
    """Load HMM from `xp.com` from given dir."""
    save_as = Path(save_as).expanduser()
    HMM = dill.load(open(save_as/"xp.com", "rb"))["HMM"]
    return HMM


def load_xps(save_as):
    """Load `xps` (as a `list`) from given dir."""
    save_as = Path(save_as).expanduser()
    files = [d/"xp" for d in uplink.list_job_dirs(save_as)]
    if not files:
        raise FileNotFoundError(f"No results found at {save_as}.")

    def load_any(filepath):
        """Load any/all `xp's` from `filepath`."""
        with open(filepath, "rb") as F:
            # If experiment crashed, then xp will be empty
            try:
                data = dill.load(F)
            except EOFError:
                return []
            # Always return list
            try:
                return data["xps"]
            except KeyError:
                return [data["xp"]]

    print("Loading %d files from %s" % (len(files), save_as))
    xps = []  # NB: progbar wont clean up properly w/ list compr.
    for f in tqdm(files, desc="Loading"):
        xps.extend(load_any(f))

    if len(xps) == 0:
        raise RuntimeError("No completed experiments found.")
    elif len(xps) < len(files):
        print(len(files)-len(xps), "files could not be loaded,",
              "presumably because their respective jobs crashed.")

    return xps


def save_xps(xps, save_as, nDir=100):
    """Save `xps` (list of `xp`s) in `nDir` subfolders: `save_as/i`.

    Example
    -------
    Rename attr. `n_iter` to `nIter` in some saved data:

    ```py
    proj_name = "Stein"
    dd = rc.dirs.data / proj_name
    save_as = dd / "run_2020-09-22__19:36:13"

    for save_as in dd.iterdir():
        save_as = dd / save_as

        xps = load_xps(save_as)
        HMM = load_HMM(save_as)

        for xp in xps:
            if hasattr(xp,"n_iter"):
                xp.nIter = xp.n_iter
                del xp.n_iter

        overwrite_xps(xps, save_as)
    ```
    """
    save_as = Path(save_as).expanduser()
    save_as.mkdir(parents=False, exist_ok=False)

    n = int(len(xps) // nDir) + 1
    splitting = [xps[i:i + n] for i in range(0, len(xps), n)]
    for i, sub_xps in enumerate(tqdm(splitting, desc="Saving")):
        if len(sub_xps):
            iDir = save_as / str(i)
            os.mkdir(iDir)
            with open(iDir/"xp", "wb") as F:
                dill.dump({'xps': sub_xps}, F)


def overwrite_xps(xps, save_as, nDir=100):
    """Save xps in save_as, but safely (by first saving to tmp)."""
    save_xps(xps, save_as/"tmp", nDir)

    # Delete
    for d in tqdm(uplink.list_job_dirs(save_as),
                  desc="Deleting old"):
        shutil.rmtree(d)

    # Mv up from tmp/ -- goes quick, coz there are not many.
    for d in os.listdir(save_as/"tmp"):
        shutil.move(save_as/"tmp"/d, save_as/d)

    shutil.rmtree(save_as/"tmp")


def reduce_inodes(save_as, nDir=100):
    """Reduce the number of `xp` dirs.

    Done by packing multiple `xp`s into lists (`xps`).
    This reduces the **number** of files (inodes) on the system, which is limited.

    It also deletes files "xp.var" and "out",
    whose main content tends to be the printed progbar.
    This probably leads to some reduced loading time.

    FAQ: Why isn't the default for `nDir` simply 1?
    So that we can get a progressbar when loading.
    """
    overwrite_xps(load_xps(save_as), save_as, nDir)
    
class NetcdfIO:
    
    def __init__(self, file_path):
        self.format = 'NETCDF4'
        self.file_path = os.path.abspath(file_path) 
        self.file_name = os.path.splitext(os.path.basename(file_path))[0]
        self.float_type = 'f8'
        self.int_type = 'i4'
        
    def create_file(self):
        from datetime import datetime, timedelta
        file_dir = os.path.dirname(self.file_path)
        
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
        
        if os.path.isfile(self.file_path):
            os.remove(self.file_path)
            
        with self.istream as nc:
           nc.creation_time = datetime.now().strftime('%Y-%m-%d %h:%m:%S')
    
    @property   
    def istream(self):
        if os.path.isfile(self.file_path): 
            return Dataset(self.file_path, 'a', format=self.format)
        else:
            return Dataset(self.file_path, 'w', format=self.format)
    
    @property 
    def ostream(self):
        return Dataset(self.file_path, 'r', format=self.format)
           
    def create_dims(self, HMM, N):
        self.create_dim_time(HMM.tseq)
        self.create_dim_state(HMM)
        self.create_dim_obs(HMM)
        if N>0:
            self.create_dim_ens(N)
        
    def create_dim_ens(self, N):                
        with self.istream as nc:
            nc.createDimension('N', N)
            
            window = nc.createVariable('window', self.int_type, ('time',))
            window.long_name = ("Index of DA window to which forecast belongs. "
                                "Index matches index of last analysis prior to forecast.")
            
            Efor = nc.createVariable('forecast', self.float_type, ('time','N','M'))
            Efor.long_name = "Ensemble of forecast states at different times."
            
            Eana = nc.createVariable('analysis', self.float_type, ('timeo','N','M'))
            Eana.long_name = "Ensemble of analysis states at different times."
        
    def create_dim_time(self, tseq):              
        with self.istream as nc:
            for var in ['K','Ko','T','BurnIn','dto','dt']:
                setattr(nc, var, getattr(tseq, var))
            
            nc.createDimension('time', tseq.K+1)
            time = nc.createVariable('time', self.float_type, ('time',))
            self.time = np.arange(0, tseq.K+1, dtype=float) * tseq.dt 
            time[:] = self.time
            time.long_name = "Times at which state values are available."
            
            nc.createDimension('timeo', tseq.Ko+1)
            timeo = nc.createVariable('timeo', self.float_type, ('timeo',))
            self.timeo = np.arange(0, tseq.Ko+1, dtype=float) * tseq.dto + tseq.BurnIn 
            self.timeo = self.timeo + max(1., np.ceil(tseq.BurnIn/tseq.dto)) * tseq.dto
            timeo[:] = self.timeo
            time.long_name = "Times at which observations are available."
            
            
    def create_dim_state(self, HMM):     
        with self.istream as nc:
            M = HMM.Dyn.M
            nc.createDimension('M', M)
            
            position = nc.createVariable('position', self.float_type, ('M',))
            if hasattr(HMM,'coordinates'):
                position[:] = HMM.coordinates
            else:
                position[:] = np.arange(0, M, dtype=float)
            position.long_name = "Spatial position of grid points."
                
            field_index = nc.createVariable('field_index', self.int_type, ('M',))
            field_names = []
            if hasattr(HMM,'sectors'):
                for no, key in enumerate(HMM.sectors):
                    field_index[HMM.sectors[key]] = no 
                    field_names.append(key)
            else:
                field_index[:] = np.zeros((M,))
            field_index.long_name = "Physical field represented by value in state vector."
              
            nc.createDimension('fields', len(field_names))
            field_name = nc.createVariable('field_name', 'S128', ('fields',))
            field_name.long_name = "Name of physical field associated with field_index."
            for no,name in enumerate(field_names):
                field_name[no] = name                
            
            xx = nc.createVariable('xx', self.float_type, ('time','M'))
            xx.long_name = "State of truth run at different times."
            
    def create_dim_obs(self, HMM):       
        x0 = np.zeros((HMM.Dyn.M,), dtype=float)
        self.Mo = np.array([np.size(HMM.Obs(x0,t)) for t in self.time], dtype=int)
        
        with self.istream as nc:
            nc.createDimension('Mo', max(self.Mo))             
            
            yy = nc.createVariable('yy', self.float_type, ('timeo','Mo'))
            yy.long_name = "Collection of observations at different times."
            
            Mo = nc.createVariable('Mo', self.int_type, ('timeo',))
            Mo.long_name = "Number of observations at different times."
            
            
    def write_truth(self, xx, yy):        
        with self.istream as nc:
            nc['xx'][:,:] = np.array(xx, dtype=float) 
            for no,y in enumerate(yy):
                nc['Mo'][no] = len(y)
                nc['yy'][no,:len(y)] = y
                
    def read_truth(self):
        with self.ostream as nc:
            xx = np.array(nc['xx'][:,:], dtype=float) 
            ya = np.array(nc['yy'][:,:], dtype=float) 
            Mo = np.array(nc['Mo'][:], dtype=int)
            
        yy = []
        for ko,m in enumerate(Mo):
            yy.append(ya[ko,:m])
        
        return xx, yy
                
    def write_forecast(self, it, E): 
        with Dataset(self.file_path, 'a', format=self.format) as nc:
            if len(self.timeo)==0 or self.time[it] <= min(self.timeo):
                nc['window'][it] = -1
            else:
                nc['window'][it] =np.argmax(self.timeo<self.time[it])
            nc['forecast'][it,:,:] = np.array(E, dtype=float)
            
    def write_analysis(self, it, E):            
        with Dataset(self.file_path, 'a', format=self.format) as nc:
            nc['analysis'][it,:,:]=np.array(E, dtype=float) 
            
class SaveXP:
    import dill, os
    
    def __init__(self, save_as):
        self.save_as = os.path.abspath(save_as) 
        
        
    def create_file(self):
        fdir = os.path.dirname(self.save_as)
        
        if not os.path.isdir(fdir):
            os.mkdir(fdir)
        
        if os.path.isfile(self.save_as):
            os.remove(self.save_as)
        
    def save_truth(self, HMM, xx, yy):
        keys = ['K','Ko','T','BurnIn','dto','dt']
        
        tseq_obj = HMM.tseq
        
        truth = {'content':'truth'}
        for key in keys:
            truth[key] = getattr(tseq_obj, key)
            
        truth['xx'] = xx
        truth['yy'] = yy        
            
        with open(self.save_as,'ab+') as stream:
            dill.dump(truth, stream)
            
    def save_forecast(self, E, k, ko):
        ensemble = {'content':'forecast','k':k,'ko':ko,'E':E}
        
        with open(self.save_as,'ab+') as stream:
            dill.dump(ensemble, stream)
            
    def save_analysis(self, E, k, ko=None):
        ensemble = {'content':'analysis','k':k,'ko':ko,'E':E}
        
        with open(self.save_as,'ab+') as stream:
            dill.dump(ensemble, stream)
            
    def load(self, contents=None, kk=None, kko=None):
        output = []
        
        with open(self.save_as,'rb') as stream:
            while True:
                try:
                    data = dill.load(stream)
                except EOFError:
                    break 
                
                if contents is not None:
                    data = self.filter(data, 'contents', contents)
                if kk is not None:
                    data = self.filter(data, 'kk', kk)
                if kko is not None:
                    data = self.filter(data, 'kko', kko)
                    
                output.append(data)
                
        return output
    
    def filter(self, data, key, valid_values):
        if not hasattr(valid_values, '__iter__'):
            valid_values = [valid_values]
            
        if key not in data:
            return data 
        elif data[key] in valid_values:
            return data 
        else:
            return []
    
    def create_stats(self, xp, HMM):
        from dapper.stats import Stats 
        
        stats = None
        read_next = True
        queue = []
        
        with open(self.save_as,'rb') as stream:
            while read_next:
                try:
                    data=dill.load(stream)
                except EOFError:
                    break
                
                if data['content']=='truth':
                    stats = Stats(xp, HMM, data['xx'], data['yy'])
                

                if data['content']=='forecast':
                    k, ko = data['k'], None
                    stats.assess(k, ko, E=data['E'])
                    
                
        return stats
                    
                
                
                    
                
                
                        
            
            
        
