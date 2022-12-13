# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 10:48:12 2022

@author: satya
"""
#Import Packages

import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

#Import Modules
from BatteryDataSOH import BatteryDataSOHC

batdata = BatteryDataSOHC()

## Battery 5
bdataname='B0005.mat'
fdataname='B0005'
SOH_typical_005 = batdata.batterydatasoh(bdataname, fdataname)

## Battery 6
bdataname='B0006.mat'
fdataname='B0006'
SOH_typical_006 = batdata.batterydatasoh(bdataname, fdataname)

## Battery 7
bdataname='B0007.mat'
fdataname='B0007'
SOH_typical_007 = batdata.batterydatasoh(bdataname, fdataname)

## Battery 7
bdataname='B0018.mat'
fdataname='B0018'
SOH_typical_018 = batdata.batterydatasoh(bdataname, fdataname)

#SOH Determination
linestyle_list=['k-','b-o','g-s','r-*','m-x','y-D','k-v','b-+'] # list of basic colors
fig = plt.figure(401)
plt.plot(range(len(SOH_typical_005)),SOH_typical_005,linestyle_list[0],linewidth=2,label= 'Battery5')
plt.plot(range(len(SOH_typical_006)),SOH_typical_006,linestyle_list[1],linewidth=2,label= 'Battery6')
plt.plot(range(len(SOH_typical_007)),SOH_typical_007,linestyle_list[2],linewidth=2,label= 'Battery7')
plt.plot(range(len(SOH_typical_018)),SOH_typical_018,linestyle_list[3],linewidth=2,label= 'Battery18')
plt.xlabel('Cycle #')
plt.ylabel('SOH')
plt.legend()
plt.grid()
plt.show