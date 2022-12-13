# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:58:09 2022

@author: RaghavS
"""
#Import Packages

import numpy as np
import scipy.io as sp
import pandas as pd
import scipy.stats
import seaborn as sns
import math
import matplotlib.pyplot as plt
import matplotlib.colors

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.optim as optimiz


from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool1d
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

#Import Modules
from DataPrep import DataPrepC


#Step 1: Dataset Loading and Interpretation, Cleaning

## Program Begins
bdataname='B0005.mat'
fdataname='B0005'
wat1 = sp.loadmat(bdataname)
wat2 = wat1[fdataname][0]['cycle'][0]

Fdataprepc = DataPrepC() 

# initializing variables
d_ambtemp={}
d_capac={}
current_cur={}
j=0;

C_usable = {}
C_t = {}
SOC_r = {}
i_c =1.5    #charge current
ctime_zero={}
dtime_zero=0
time_cur={}
i_d =2      #discharge 

linestyle_list=['k-','b-o','g-s','r-*','m-x','y-D','k-v','b-+'] # list of basic colors

#Sampling Point Determination
samplingpt=20
sampling_array={}



#Capacity Determination
m=0;
n=0;
C_usable_data = {}
SOH_typical={}
d_capac={}
d_discharge_idx={}
c_charge_idx={}
a_c_charge_idx={}
d_discharge_flag=False
for i in range(10, np.size(wat2)-2):
    stype = np.array2string(wat1[fdataname][0]['cycle'][0]['type'][0][i])
    if i==10:
        d_discharge_flag=True
    if stype == "['charge']" and d_discharge_flag==True:
        c_charge_idx[n]=i
        d_discharge_flag=False
        n=n+1
        
    if stype == "['discharge']":
        dcapac = float(np.array(wat1[fdataname][0]['cycle'][0]['data'][0][i]['Capacity'][0]))
        d_capac[m]=dcapac
        d_discharge_idx[m]=i
        d_discharge_flag=True
        m=m+1
#d_scapac[m]=d_capac 
a_c_charge_idx = np.array(list(c_charge_idx.items()))[:,1]
a_d_discharge_idx = np.array(list(d_discharge_idx.items()))
a_dcapac = np.array(list(d_capac.items())) 
SOH_typical = a_dcapac[:,1]/a_dcapac[0,1]

#SOH Determination
fig = plt.figure(4)
ax = plt.axes()
ax.plot(range(len(SOH_typical)),SOH_typical,linewidth=2)
plt.xlabel('Cycle #')
plt.ylabel('SOH')
plt.grid()
plt.show


# Sample Point Determination at Regular Intervals
dcapac_samp_intv = [1.0, 0.95,0.90,0.85,0.80,0.75,0.70]
#dcapac_samp_idx={}
d_discharge_idx_samp={}
c_charge_idx_samp = {}
for i in range(len(dcapac_samp_intv)):
    dcapac_samp=Fdataprepc.find_nearest(SOH_typical, value=dcapac_samp_intv[i])
    dcapac_samp_idx=np.where(SOH_typical==dcapac_samp)[0][0]
    d_discharge_idx_samp[i] = d_discharge_idx[dcapac_samp_idx]
    c_charge_idx_samp[i] = Fdataprepc.find_nearest(a_c_charge_idx, value=d_discharge_idx_samp[i])

a_c_charge_idx_samp = np.array(list(c_charge_idx_samp.items()))[:,1] 


a_c_charge_idx_run={}
a_c_charge_idx_bool = True #set true for full run and false for sample run

if a_c_charge_idx_bool:
    a_c_charge_idx_run=a_c_charge_idx
else:
    a_c_charge_idx_run=a_c_charge_idx_samp

#matrix initialization
sptime_time = {}
sptime_current = {}
sptime_volt = {}
sptime_deltaT ={}
spsoh_time = {}
spsoh_current = {}
spsoh_volt = {}
spsoh_deltaT ={}
sptime_kk={}

sptime_time_samp_precapacity={}
sptime_current_samp_precapacity={}
sptime_volt_samp_precapacity={}
sptime_deltaT_samp_precapacity={}
spsoh_time_samp_precapacity={}
spsoh_current_samp_precapacity={}
spsoh_volt_samp_precapacity={}
spsoh_deltaT_samp_precapacity={}
 
C_usable_sp={}

#index initialization
m=0;
j=0;
j1=0;
for i in range(np.size(wat2)):  
    SOC_k_t = {}
    SOC_k_t[0] = 0
    
    #C_Rated
    if i==0:
        C_usable = float(np.array(wat1[fdataname][0]['cycle'][0]['data'][0][1]['Capacity'][0]))
        
    stype = np.array2string(wat1[fdataname][0]['cycle'][0]['type'][0][i])
    if stype == "['charge']":

        c_ambtemp = float(wat1[fdataname][0]['cycle'][0]['ambient_temperature'][0][i])
        c_time = np.array(wat1[fdataname][0]['cycle'][0]['data'][0][i]['Time'][0][0][0])
        c_voltm = np.array(wat1[fdataname][0]['cycle'][0]['data'][0][i]['Voltage_measured'][0][0][0])
        c_currentm = np.array(wat1[fdataname][0]['cycle'][0]['data'][0][i]['Current_measured'][0][0][0])
        c_voltagec = np.array(wat1[fdataname][0]['cycle'][0]['data'][0][i]['Voltage_charge'][0][0][0])
        c_tempm = np.array(wat1[fdataname][0]['cycle'][0]['data'][0][i]['Temperature_measured'][0][0][0])
        
        #SOC - Relative Determination
        for k in range(1,np.size(c_currentm)):
            SOC_k_t[k] = SOC_k_t[k-1] + (1/C_usable)*(c_currentm[k]*((c_time[k]-c_time[k-1])/(60*60)))
            if SOC_k_t[k]<0:
                SOC_k_t[k]=0
            if SOC_k_t[k]>1:
                SOC_k_t[k]=1            
        
        a_SOC_k_t = np.array(list(SOC_k_t.items()))
        a_SOC_k_tp = a_SOC_k_t[:,1]*100
                  
        #Temperature estimation for Charge Cycle
        delta_Tmk = c_tempm - min(c_tempm)
        
        #Sampling Determination based on data size
        if (samplingpt < np.size(c_time)) and (samplingpt!=0):
            sampling_array= np.array(list(range(0,np.size(c_time),int(np.size(c_time)/samplingpt))))
        else:
            sampling_array= np.array(list(range(0,np.size(c_time),int(np.size(c_time)/np.size(c_time)))))

        #Sampling Determination based on c_time data
        if samplingpt!=0:
            ctime_sampling_array_data= np.array(list(range(int(round(np.min(c_time))),int(round(np.max(c_time))),int(round((np.max(c_time)-np.min(c_time))/samplingpt)))))            
            if (len(ctime_sampling_array_data)>samplingpt):
                ctime_sampling_array_data=ctime_sampling_array_data[0:samplingpt]
            ctime_samp={}
            ctime_samp_idx={}
            for k in range(len(ctime_sampling_array_data)):
                ctime_samp[k]=Fdataprepc.find_nearest(c_time, value=ctime_sampling_array_data[k])
                ctime_samp_idx[k]=np.where(c_time==Fdataprepc.find_nearest(c_time, value=ctime_sampling_array_data[k]))[0][0]
        else:
            ctime_samp = c_time
            ctime_samp_idx = range(0,np.size(c_time))
            
        #Sampling Determination based on SOH data
        if samplingpt!=0 and int(round(np.max(a_SOC_k_tp)))>0 and np.max(a_SOC_k_tp)>samplingpt:
            csoh_sampling_array_data= np.array(list(range(int(round(np.min(a_SOC_k_tp))),int(round(np.max(a_SOC_k_tp))),int(round((np.max(a_SOC_k_tp)-np.min(a_SOC_k_tp))/samplingpt)))))
            if (len(csoh_sampling_array_data)>samplingpt):
                csoh_sampling_array_data=csoh_sampling_array_data[0:samplingpt]
            csoh_samp={}
            csoh_samp_idx={}
            for k in range(len(csoh_sampling_array_data)):
                csoh_samp[k]=Fdataprepc.find_nearest(a_SOC_k_tp, value=csoh_sampling_array_data[k])
                csoh_samp_idx[k]=np.where(a_SOC_k_tp==Fdataprepc.find_nearest(a_SOC_k_tp, value=csoh_sampling_array_data[k]))[0][0]
        else:
            csoh_samp = a_SOC_k_tp
            csoh_samp_idx = range(0,np.size(a_SOC_k_tp))

        #Graph based on Time
        if i in a_c_charge_idx_run:
        #if i==10:
            #print(i)
            a_ctime_samp={}
            a_ctime_samp_idx={}
            a_ctime_currentm={}
            a_ctime_voltm = {}
            a_ctime_delta_Tmk = {}
            
            a_ctime_samp,a_ctime_voltm
            
            a_ctime_samp = np.array(list(ctime_samp.items()))[:,1]
            a_ctime_samp_idx = np.array(list(ctime_samp_idx.items()))[:,1]
            #a_ctime_samp_idx = np.array(list(ctime_samp_idx))
           
            a_ctime_currentm = c_currentm[a_ctime_samp_idx]
            a_ctime_voltm = c_voltm[a_ctime_samp_idx]
            a_ctime_delta_Tmk = delta_Tmk[a_ctime_samp_idx]

            a_csoh_samp = np.array(list(csoh_samp.items()))[:,1]
            a_csoh_samp_idx = np.array(list(csoh_samp_idx.items()))[:,1]
            a_csoh_currentm = c_currentm[a_csoh_samp_idx]
            a_csoh_voltm = c_voltm[a_csoh_samp_idx]
            a_csoh_delta_Tmk = delta_Tmk[a_csoh_samp_idx]

                       
            #5th term
            sp_kk={}
            sp_kk[0]=i
            for k in range(1,samplingpt):
                sp_kk[k]= sp_kk[0] + (1/(samplingpt-k))


            #Varriable assignment for Sampling Points
            sptime_time[j] = a_ctime_samp
            sptime_current[j] = a_ctime_currentm
            sptime_volt[j] = a_ctime_voltm
            sptime_deltaT[j] = a_ctime_delta_Tmk
            
            spsoh_time[j] = a_csoh_samp
            spsoh_current[j] = a_csoh_currentm
            spsoh_volt[j] = a_csoh_voltm
            spsoh_deltaT[j] = a_csoh_delta_Tmk
            
            sptime_kk[j]= np.array(list(sp_kk.items()))[:,1]
 
            
            #Sampling Point at predefined Capacity
            if i in a_c_charge_idx_samp:
                #print(i);
                sptime_time_samp_precapacity[j1]=a_ctime_samp
                sptime_current_samp_precapacity[j1] = a_ctime_currentm
                sptime_volt_samp_precapacity[j1]= a_ctime_voltm
                sptime_deltaT_samp_precapacity[j1]= a_ctime_delta_Tmk

                spsoh_time_samp_precapacity[j1]=a_csoh_samp
                spsoh_current_samp_precapacity[j1] = a_csoh_currentm
                spsoh_volt_samp_precapacity[j1]= a_csoh_voltm
                spsoh_deltaT_samp_precapacity[j1]= a_csoh_delta_Tmk
                
                j1=j1+1

            C_usable_sp[j]=C_usable                
            
            j=j+1
  
    if stype == "['discharge']":
        dambtemp = float(wat1[fdataname][0]['cycle'][0]['ambient_temperature'][0][i])
        dcapac = float(np.array(wat1[fdataname][0]['cycle'][0]['data'][0][i]['Capacity'][0]))
        d_ambtemp[m]=dambtemp
        d_capac[m]=dcapac
        m=m+1
        
        d_time = np.array(wat1[fdataname][0]['cycle'][0]['data'][0][i]['Time'][0][0][0])
        d_voltm = np.array(wat1[fdataname][0]['cycle'][0]['data'][0][i]['Voltage_measured'][0][0][0])
        d_currentm = np.array(wat1[fdataname][0]['cycle'][0]['data'][0][i]['Current_measured'][0][0][0])
        d_voltagel = np.array(wat1[fdataname][0]['cycle'][0]['data'][0][i]['Voltage_load'][0][0][0])
        d_currentl = np.array(wat1[fdataname][0]['cycle'][0]['data'][0][i]['Current_load'][0][0][0])
        d_tempm = np.array(wat1[fdataname][0]['cycle'][0]['data'][0][i]['Temperature_measured'][0][0][0])
        
        d_check1 = 0
        time_cutoff = None
        volt_cutoff = None
        for k in range(np.size(d_voltm)):
            if d_voltm[k]<2.7 and d_check1==0:
                d_check1=1
                dtime_zero = d_time[0]
                time_cutoff = d_time[k]
                volt_cutoff = d_voltm[k]

        #Relative SOC Determination
        C_usable = i_d*((time_cutoff-dtime_zero)/(60*60))
        
        #Temperature estimation for Discharge Cycle
        delta_Tmk = d_tempm - min(d_tempm)
       
    C_usable_data[m]=C_usable

a_C_usable_sp = np.array(list(C_usable_sp.items()))[:,1]


a_sptime_time={}
a_sptime_volt={}
a_sptime_current={}
a_sptime_deltaT={}
a_sptime_kk={}

a_spsoh_time={}
a_spsoh_volt={}
a_spsoh_current={}
a_spsoh_deltaT={}

a_sptime_time = np.array(list(sptime_time.items()))[:,1]
a_sptime_volt = np.array(list(sptime_volt.items()))[:,1]
a_sptime_current = np.array(list(sptime_current.items()))[:,1]
a_sptime_deltaT = np.array(list(sptime_deltaT.items()))[:,1]

a_spsoh_time = np.array(list(spsoh_time.items()))[:,1]
a_spsoh_volt = np.array(list(spsoh_volt.items()))[:,1]
a_spsoh_current = np.array(list(spsoh_current.items()))[:,1]
a_spsoh_deltaT = np.array(list(spsoh_deltaT.items()))[:,1]

a_sptime_kk = np.array(list(sptime_kk.items()))[:,1]

# Find min and max for each array for Normalization
a_sptime_time_min = Fdataprepc.MinData(a_sptime_time)
a_sptime_time_max = Fdataprepc.MaxData(a_sptime_time)
a_sptime_volt_min = Fdataprepc.MinData(a_sptime_volt)
a_sptime_volt_max = Fdataprepc.MaxData(a_sptime_volt)
a_sptime_current_min = Fdataprepc.MinData(a_sptime_current)
a_sptime_current_max = Fdataprepc.MaxData(a_sptime_current)
a_sptime_deltaT_min = Fdataprepc.MinData(a_sptime_deltaT)
a_sptime_deltaT_max = Fdataprepc.MaxData(a_sptime_deltaT)
#soh 
a_spsoh_time_min = Fdataprepc.MinData(a_spsoh_time)
a_spsoh_time_max = Fdataprepc.MaxData(a_spsoh_time)
a_spsoh_volt_min = Fdataprepc.MinData(a_spsoh_volt)
a_spsoh_volt_max = Fdataprepc.MaxData(a_spsoh_volt)
a_spsoh_current_min = Fdataprepc.MinData(a_spsoh_current)
a_spsoh_current_max = Fdataprepc.MaxData(a_spsoh_current)
a_spsoh_deltaT_min = Fdataprepc.MinData(a_spsoh_deltaT)
a_spsoh_deltaT_max = Fdataprepc.MaxData(a_spsoh_deltaT)
#k
a_sptime_kk_min = Fdataprepc.MinData(a_sptime_kk)
a_sptime_kk_max = Fdataprepc.MaxData(a_sptime_kk)

#Step 2: Dataset Evaluation, Normalization

#2A: Correlation 

#Initialization
corr_pearson_r_sptime_volt={}
corr_pearson_r_sptime_current={}
corr_pearson_r_sptime_deltaT={}
corr_pearson_r_spsoh_volt={}
corr_pearson_r_spsoh_current={}
corr_pearson_r_spsoh_deltaT={}

#Pearson Correlation for Time Based
corr_pearson_r_sptime_volt = Fdataprepc.corr_pearsonr(samplingpt,sptime_volt, a_C_usable_sp)
corr_pearson_r_sptime_current = Fdataprepc.corr_pearsonr(samplingpt,sptime_current, a_C_usable_sp)
corr_pearson_r_sptime_deltaT = Fdataprepc.corr_pearsonr(samplingpt,sptime_deltaT, a_C_usable_sp)

#Pearson Correlation for SOH Based
corr_pearson_r_spsoh_volt = Fdataprepc.corr_pearsonr(samplingpt,spsoh_volt, a_C_usable_sp)
corr_pearson_r_spsoh_current = Fdataprepc.corr_pearsonr(samplingpt,spsoh_current, a_C_usable_sp)
corr_pearson_r_spsoh_deltaT = Fdataprepc.corr_pearsonr(samplingpt,spsoh_deltaT, a_C_usable_sp)

batt_data={}
sns.set()
batt_data = [np.array(list(corr_pearson_r_sptime_volt.items()))[:,1],np.array(list(corr_pearson_r_sptime_current.items()))[:,1],np.array(list(corr_pearson_r_sptime_deltaT.items()))[:,1], np.array(list(corr_pearson_r_spsoh_volt.items()))[:,1],np.array(list(corr_pearson_r_spsoh_current.items()))[:,1],np.array(list(corr_pearson_r_spsoh_deltaT.items()))[:,1]] 
fig = plt.figure(301)
fig.set_size_inches(8, 4)
x_axis_labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] # labels for x-axis
y_axis_labels = ['V','I','T','V','I','T'] # labels for y-axis
ax = sns.heatmap(batt_data, vmin=-1, vmax=1,xticklabels=x_axis_labels, yticklabels=y_axis_labels)
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([-1,-0.75,-0.5,-0.25, 0, 0, 0.25, 0.5, 0.75, 1.0])
plt.tight_layout()
group_edges = np.array([0, 3,6])
group_labels = ['Time based', 'SOC Based']
ax.hlines(group_edges, np.zeros(len(group_edges)), np.zeros(len(group_edges)) - 0.12,color='navy', lw=2, clip_on=False, transform=ax.get_yaxis_transform())
for label, b0, b1 in zip(group_labels, group_edges[:-1], group_edges[1:]):
    ax.text(-0.12, (b0 + b1) / 2, label, color='navy', fontsize=12, ha='left', va='center',rotation=90, transform=ax.get_yaxis_transform())
plt.show()

#Absolute Average
corr_pearson_r_sptime_volt_avg = np.average(abs(batt_data[0]))
corr_pearson_r_sptime_current_avg = np.average(abs(batt_data[1]))
corr_pearson_r_sptime_deltaT_avg = np.average(abs(batt_data[2]))
corr_pearson_r_sptime_avg = (1/3)*(corr_pearson_r_sptime_volt_avg+corr_pearson_r_sptime_current_avg+corr_pearson_r_sptime_deltaT_avg)
corr_pearson_r_spsoh_volt_avg = np.average(abs(batt_data[3]))
corr_pearson_r_spsoh_current_avg = np.average(abs(batt_data[4]))
corr_pearson_r_spsoh_deltaT_avg = np.average(abs(batt_data[5]))
corr_pearson_r_spsoh_avg = (1/3)*(corr_pearson_r_spsoh_volt_avg+corr_pearson_r_spsoh_current_avg+corr_pearson_r_spsoh_deltaT_avg)

#corr_pearson_r, corr_pearson_pvalue = scipy.stats.spearmanr(x, y)   # Spearman's rho
#scipy.stats.kendalltau(x, y).correlation  # Kendall's tau



#2B: Normalization : MinMaxNormaliation for the selected set
sptime_volt_norm={}
sptime_current_norm={}
sptime_deltaT_norm={}
sptime_time_norm={}

spsoh_volt_norm={}
spsoh_current_norm={}
spsoh_deltaT_norm={}
spsoh_time_norm={}

sptime_kk_norm={}

#Approach 1
'''
for i in range(len(sptime_time)):
    sptime_time_norm[i] = Fdataprepc.NormalizeData(sptime_time[i])
    sptime_volt_norm[i] = Fdataprepc.NormalizeData(sptime_volt[i])
    sptime_current_norm[i] = Fdataprepc.NormalizeData(sptime_current[i])
    sptime_deltaT_norm[i] = Fdataprepc.NormalizeData(sptime_deltaT[i])
    
    spsoh_time_norm[i] = Fdataprepc.NormalizeData(spsoh_time[i])
    spsoh_volt_norm[i] = Fdataprepc.NormalizeData(spsoh_volt[i])
    spsoh_current_norm[i] = Fdataprepc.NormalizeData(spsoh_current[i])
    spsoh_deltaT_norm[i] = Fdataprepc.NormalizeData(spsoh_deltaT[i])
'''
#Approach 2
for i in range(len(sptime_time)):
    sptime_time_norm[i] = Fdataprepc.NormalizeDataMinMax(sptime_time[i],a_sptime_time_min,a_sptime_time_max)
    sptime_volt_norm[i] = Fdataprepc.NormalizeDataMinMax(sptime_volt[i],a_sptime_volt_min,a_sptime_volt_max)
    sptime_current_norm[i] = Fdataprepc.NormalizeDataMinMax(sptime_current[i],a_sptime_current_min,a_sptime_current_max)
    sptime_deltaT_norm[i] = Fdataprepc.NormalizeDataMinMax(sptime_deltaT[i],a_sptime_deltaT_min,a_sptime_deltaT_max)
    
    spsoh_time_norm[i] = Fdataprepc.NormalizeDataMinMax(spsoh_time[i],a_spsoh_time_min,a_spsoh_time_max)
    spsoh_volt_norm[i] = Fdataprepc.NormalizeDataMinMax(spsoh_volt[i],a_spsoh_volt_min,a_spsoh_volt_max)
    spsoh_current_norm[i] = Fdataprepc.NormalizeDataMinMax(spsoh_current[i],a_spsoh_current_min,a_spsoh_current_max)
    spsoh_deltaT_norm[i] = Fdataprepc.NormalizeDataMinMax(spsoh_deltaT[i],a_spsoh_deltaT_min,a_spsoh_deltaT_max)
  
    sptime_kk_norm[i]=Fdataprepc.NormalizeDataMinMax(sptime_kk[i],a_sptime_kk_min,a_sptime_kk_max)
    
#Output Normalization
a_C_usable_sp_norm = Fdataprepc.NormalizeData(a_C_usable_sp)


#Step 3: Make Dataset Iterable
a_sptime_time_norm = np.array(list(sptime_time_norm.items()))[:,1]
a_sptime_volt_norm = np.array(list(sptime_volt_norm.items()))[:,1]
a_sptime_current_norm = np.array(list(sptime_current_norm.items()))[:,1]
a_sptime_deltaT_norm = np.array(list(sptime_deltaT_norm.items()))[:,1]

a_spsoh_time_norm = np.array(list(spsoh_time_norm.items()))[:,1]
a_spsoh_volt_norm = np.array(list(spsoh_volt_norm.items()))[:,1]
a_spsoh_current_norm = np.array(list(spsoh_current_norm.items()))[:,1]
a_spsoh_deltaT_norm = np.array(list(spsoh_deltaT_norm.items()))[:,1]

a_sptime_kk_norm = np.array(list(sptime_kk_norm.items()))[:,1]

X_train={}
# Trainset and validation
trainset=round(0.7*len(a_sptime_time_norm))  #50% assumed train.
for i in range(trainset):
    #X_train[i]=[np.vstack(a_sptime_time_norm[i]).reshape(-1), np.vstack(a_sptime_volt_norm[i]).reshape(-1) , np.vstack(a_sptime_current_norm[i]).reshape(-1), np.vstack(a_sptime_deltaT_norm[i]).reshape(-1)]
    X_train[i]=[np.vstack(a_spsoh_time_norm[i]).reshape(-1), np.vstack(a_spsoh_volt_norm[i]).reshape(-1) , np.vstack(a_spsoh_current_norm[i]).reshape(-1), np.vstack(a_spsoh_deltaT_norm[i]).reshape(-1), np.vstack(a_sptime_kk_norm[i]).reshape(-1)]
   
y_train = a_C_usable_sp_norm[0: trainset]

X_train, X_val, y_train, y_val = train_test_split(X_train,  y_train, test_size=0.2)
X_train, y_train, X_val, y_val = map(torch.tensor, (X_train, y_train, X_val,  y_val))

X_train = X_train.float()
y_train = y_train.float()
X_val = X_val.float()
y_val = y_val.float()

#Step 4: Create Model Class
class ConvNeuralNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = Conv1d(5,20,kernel_size=3)
        #self.relu1 = ReLU()
        #self.maxpool1 = MaxPool1d(kernel_size=2, stride=2)
        
		# initialize second set of CONV => RELU => POOL layers
        #self.conv2 = Conv1d(20,40,kernel_size=4)
        #self.relu2 = ReLU()
        #self.maxpool2 = MaxPool1d(kernel_size=2, stride=2)
        
		# initialize first (and only) set of FC => RELU layers
        #self.fc1 = Linear(in_features=40, out_features=1)
        #self.relu3 = ReLU()
		# initialize our softmax classifier
        #self.fc2 = Linear(500, 1)
        #self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
        #x = self.conv1(x)
        output  = self.conv1(x)
        #x = self.relu1(x)
        
        #x = self.maxpool1(x)
        #output  = self.maxpool1(x) 
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
        #x = self.conv2(x)
        #x = self.relu2(x)
        #x = self.maxpool2(x)
		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
        #x = flatten(x, 1)
        #x = self.fc1(x)
        #x = self.relu3(x)
		# pass the output to our softmax classifier to get our output
		# predictions
        #x = self.fc2(x)
        #output = self.logSoftmax(x)
		# return the output predictions
        return output

    def predict(self, X):
        Y_pred = self.forward(X)
        return Y_pred


def accuracy(y_hat, y):
   pred = torch.argmax(y_hat, dim=1)
   return (pred == y).float().mean()

#def fit(x, y, model, opt, loss_fn, epochs = 100):



#return loss.item()
#Step 5: Instantiate Model Class
model = ConvNeuralNetModel()
device = torch.device("cuda")

#X_train=X_train.to(device)
#y_train=y_train.to(device)
#X_val=X_val.to(device)
#y_val=y_val.to(device)
#model.to(device)

#Step 6: Instantiate Loss Class
lossFn = nn.NLLLoss()

#Step 7: Instantiate Optimizer Class
learning_rate = 0.001
optimizer = optimiz.Adam(model.parameters(),lr=learning_rate)
# initialize a dictionary to store training history
H = {"train_loss": [], "train_acc": [], "val_loss": [],	"val_acc": []}
#print('Final loss', fit(X_train, y_train, model, optimizer , criterion))


#Step 8: Train Model
# initialize the train, validation, and test data loaders
BATCH_SIZE=90
trainDataLoader = DataLoader((X_train, y_train),batch_size=BATCH_SIZE)
valDataLoader = DataLoader((X_val, y_val), batch_size=BATCH_SIZE)


# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

#Training your first Convolutional Neural Network (CNN)
EPOCHS = 10
for e in range(1, EPOCHS):
	# set the model in training mode
    print('fg',e)
    model.train()
	# initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
	# initialize the number of correct predictions in the training
	# and validation step
    trainCorrect = 0
    valCorrect = 0
	# loop over the training set
    for (x, y) in trainDataLoader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFn(pred, y)
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
	# switch off autograd for evaluation
    with torch.no_grad():
		# set the model in evaluation mode
        model.eval()
		# loop over the validation set
        for (x, y) in valDataLoader:
			# send the input to the device
            (x, y) = (x.to(device), y.to(device))
			# make the predictions and calculate the validation loss
            pred = model(x)
            totalValLoss += lossFn(pred, y)
			# calculate the number of correct predictions
            valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()


	# calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps
	# calculate the training and validation accuracy
    trainCorrect = trainCorrect / len(trainDataLoader.dataset)
    valCorrect = valCorrect / len(valDataLoader.dataset)
	# update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["train_acc"].append(trainCorrect)
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())
    H["val_acc"].append(valCorrect)
	# print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))


    


#Step 9: Test the Model
X_test={}
X_dummy={}
y_test={}
y_dummy={}
mult_kk={}
m=0
#Remaining portion considered for test
for i in range(trainset, len(a_sptime_time_norm)):
    #X_test[m]=[np.vstack(a_sptime_time_norm[i]).reshape(-1), np.vstack(a_sptime_volt_norm[i]).reshape(-1) , np.vstack(a_sptime_current_norm[i]).reshape(-1), np.vstack(a_sptime_deltaT_norm[i]).reshape(-1)]
    X_test[m]=[np.vstack(a_spsoh_time_norm[i]).reshape(-1), np.vstack(a_spsoh_volt_norm[i]).reshape(-1) , np.vstack(a_spsoh_current_norm[i]).reshape(-1), np.vstack(a_spsoh_deltaT_norm[i]).reshape(-1), np.vstack(a_sptime_kk_norm[i]).reshape(-1) ]

    m=m+1
    
y_test = a_C_usable_sp_norm[trainset:len(a_sptime_time_norm)]
X_test, X_dummy, y_test, y_dummy = train_test_split(X_test,  y_test, test_size=0.01)

X_test = torch.tensor(X_test)
X_test = X_test.float()
X_test=X_test.to(device)

testDataLoader = DataLoader(X_test, batch_size=BATCH_SIZE)
print("[INFO] evaluating network...")
# turn off autograd for testing evaluation
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()
	
	# initialize a list to store our predictions
	preds = []
	# loop over the test set
	for (x, y) in testDataLoader:
		# send the input to the device
		x = x.to(device)
		# make the predictions and add them to the list
		pred = model(x)
		preds.extend(pred.argmax(axis=1).cpu().numpy())


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")



'''
Y_pred_val = model.predict(X_test)

#Tensor to Numpy Array
a_Y_pred_val = Y_pred_val.cpu().detach().numpy()

#Extracting Value
a_Y_pred_val_extract = {}
for i in range(trainset, len(a_sptime_time_norm)-1):
    a_Y_pred_val_extract[i-trainset]=1-(np.mean(a_sptime_kk_norm[i])*(1-np.hstack(a_Y_pred_val[i-trainset,4]).reshape(-1)))

#SOH Prediction
SOH_typical_pred={}
SOH_typical_pred=[*SOH_typical[0:trainset], *np.hstack(np.array(list(a_Y_pred_val_extract.items()))[:,1])]

plt.figure(501)
plt.plot(range(len(SOH_typical)),SOH_typical,linewidth=2)
plt.plot(range(len(SOH_typical_pred)),SOH_typical_pred,linewidth=2)
plt.xlabel('Cycle #')
plt.ylabel('SOH')
plt.grid()
plt.show

#%%

#Graph based on Time
plt.figure(101)
for i in range(len(sptime_time_samp_precapacity)):
    plt.plot(np.array(list(sptime_time_samp_precapacity.items()))[i,1],np.array(list(sptime_current_samp_precapacity.items()))[i,1], linestyle_list[i], linewidth=2, label= dcapac_samp_intv[i]*100)

plt.xlim([0,12000])
plt.ylim([0,2.0])
plt.legend()
plt.ylabel('Current (A)')
plt.xlabel('time')
plt.show    

plt.figure(102)
for i in range(len(sptime_time_samp_precapacity)):
    plt.plot(np.array(list(sptime_time_samp_precapacity.items()))[i,1],np.array(list(sptime_volt_samp_precapacity.items()))[i,1], linestyle_list[i], linewidth=2, label= dcapac_samp_intv[i]*100)

plt.xlim([0,12000])
plt.ylim([3.0,4.3])
plt.legend()
plt.ylabel('Voltage (V)')
plt.xlabel('time')
plt.show 

plt.figure(103)
for i in range(len(sptime_time_samp_precapacity)):
    plt.plot(np.array(list(sptime_time_samp_precapacity.items()))[i,1],np.array(list(sptime_deltaT_samp_precapacity.items()))[i,1], linestyle_list[i], linewidth=2, label= dcapac_samp_intv[i]*100)

plt.xlim([0,12000])
plt.ylim([0,7])
plt.legend()
plt.ylabel('delta T (degC)')
plt.xlabel('time')
plt.show      

#Graph based on SOH Relative
plt.figure(104)
for i in range(len(sptime_time_samp_precapacity)):   
    plt.plot(np.array(list(spsoh_time_samp_precapacity.items()))[i,1],np.array(list(spsoh_current_samp_precapacity.items()))[i,1], linestyle_list[i], linewidth=2, label= dcapac_samp_intv[i]*100)

plt.xlim([0,100])
plt.ylim([0,1.75])
plt.legend()
plt.ylabel('Current (A)')
plt.xlabel('SOC_Relative %')
plt.show    

plt.figure(105)
for i in range(len(sptime_time_samp_precapacity)):
    plt.plot(np.array(list(spsoh_time_samp_precapacity.items()))[i,1],np.array(list(spsoh_volt_samp_precapacity.items()))[i,1], linestyle_list[i], linewidth=2, label= dcapac_samp_intv[i]*100)

plt.xlim([0,100])
plt.ylim([3.0,4.3])
plt.legend()
plt.ylabel('Voltage (V)')
plt.xlabel('SOC_Relative %')
plt.show 

plt.figure(106)
for i in range(len(sptime_time_samp_precapacity)):
    plt.plot(np.array(list(spsoh_time_samp_precapacity.items()))[i,1],np.array(list(spsoh_deltaT_samp_precapacity.items()))[i,1], linestyle_list[i], linewidth=2, label= dcapac_samp_intv[i]*100)

plt.xlim([0,100])
plt.ylim([0,7])
plt.legend()
plt.ylabel('delta T (degC)')
plt.xlabel('SOC_Relative %')
plt.show      
      
'''




#%%
