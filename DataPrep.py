# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:43:40 2022

@author: satya
"""
#Import Packages

import numpy as np
import scipy.stats

class DataPrepC:
    # Find Nearest Value
    def find_nearest(self,array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    
    #Pearson Correlation Coefficient
    def corr_pearsonr(self,samplingpts, array_verification, array_usable):
        correl_x = {}
        corr_pearson_r={}
        
        for i in range(samplingpts):
            for j in range(len(array_verification)):
                correl_x[j] = array_verification[j][i]
            a_correl_x = np.array(list(correl_x.items()))[:,1]
            #pearson_r_matrx = np.corrcoef(array_calculated,array_verification)
            pearson_r, pearson_pvalue = scipy.stats.pearsonr(a_correl_x,array_usable)    # Pearson's r
            corr_pearson_r[i] = pearson_r
        return corr_pearson_r
    
    def NormalizeData(self,data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def MinData(self,data):
        data_min={}
        for i in range(len((data))):
            data_min[i]=np.min(data[i])
        
        return np.min(np.array(list(data_min.items()))[:,1])
    
    def MaxData(self,data):
        data_max={}
        for i in range(len((data))):
            data_max[i]=np.max(data[i])
        
        return np.max(np.array(list(data_max.items()))[:,1])
    
    def NormalizeDataMinMax(self,data,data_min, data_max):
        return (data - data_min) / (data_max - data_min)