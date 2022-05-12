#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:35:11 2022

@author: ivo
"""

from demo import *

HMM = aev_pnormal()

xps = xpList()
xps = xp_wind(xps,'test')

truth = xps[-1].save_xp.load_content('truth')
forecast = xps[-1].save_xp.load_content('forecast')
analysis = xps[-1].save_xp.load_content('analysis')
truth = xps[-1].save_xp.load_content('truth') 


def get_time(contents):
    kk = [] 
    kko = []
    E = []
    for no, content in enumerate(contents):
        k, ko, E1 = None, None, None
        if 'k' in content:
            k=content['k']
        if 'ko' in content:
            ko=content['ko']
        if 'E' in content:
            E1=content['E']
        
        kk.append(k)
        kko.append(ko)
        E.append(E1)
        
    return kk, kko,np.array(E,dtype=float)


            

