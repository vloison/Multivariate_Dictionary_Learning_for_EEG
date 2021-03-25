#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
# files handling
import pickle
from os import listdir
#from os.path import exists
#from scipy.io import loadmat
import mne


# In[ ]:


def dictionary_update(dictionary, decomposition, residual, learning_rate=0):
    
    n_kernels = len(dictionary)

    n_active_atoms = decomposition.shape[0]
    signal_len, n_dims = residual.shape
    amplitudes = decomposition[:, 0]
    offsets = decomposition[:, 1].astype(int)
    indexes = decomposition[:, 2].astype(int)
    H_sum = 0
    H_count = 0

    # Initialization
    gradient = [np.zeros_like(dictionary[i]) for i in range(n_kernels)]

    for i in range(n_active_atoms):
        k_len = dictionary[indexes[i]].shape[0]
        if k_len + offsets[i]  < signal_len:
            # Do not consider oversized atoms
            r = residual[offsets[i] : k_len + offsets[i], :]  # modif
            gradient[indexes[i]] += np.conj(amplitudes[i] * r)


    # First pass to estimate the step
    step = np.zeros((n_kernels, 1))
    for i in range(n_kernels):
        k_len = dictionary[i].shape[0]
        kernel_indexes = np.where(indexes == i)[0]
        
        #get sorted offsets
        kernel_offsets = offsets[kernel_indexes]
        sorted_kernel_offsets = np.sort(kernel_offsets)
        proper_order = np.argsort(kernel_offsets)
        
        #get sorted amplitudes
        kernel_amplitudes = amplitudes[kernel_indexes]
        sorted_kernel_amplitudes = kernel_amplitudes[proper_order]
        
        #Find self overlapping kernels
        diff_offsets = sorted_kernel_offsets[1:] - sorted_kernel_offsets[:-1]
        overlapping = np.where(diff_offsets>k_len)
        
        if np.sum(overlapping) == 0:
            H_corr = 0
        else:
            H_corr = (
                2.0
                * np.sum(
                    np.abs(kernel_amplitudes[overlapping] * kernel_amplitudes[overlapping + 1])
                    * (k_len - diff_offsets[overlapping])
                )
                / k_len
            )
        H_base = np.sum(np.abs(amplitudes[kernel_indexes]) ** 2)
        # if learning_rate+hessian_corr+hessian_base == 0.:
        if learning_rate == 0.0:
            # Gauss-Newton method if mu = 0
            step[i] = 0
        else:
            step[i] = 1.0 / (learning_rate + H_corr + H_base)
        if (H_corr + H_base) != 0:
            H_sum += H_corr + H_base
            H_count += 1

    gradient = [gradient[i] * step[i] for i in range(n_kernels)]
    
    ########
    #Dictionary update
    ########
    for i in range(len(dictionary)):
        dictionary[i] = dictionary[i] + gradient[i]


    return dictionary


def calculate_residuals(signal,dictionary,decomposition):

    recomp = np.zeros(signal.shape)
    for i in range(decomposition.shape[0]):
        y_add = recomp.copy()
        offset = decomposition[i,1].astype(int)
        idx = decomposition[i,2].astype(int)
        k_len = dictionary[idx].shape[0]
        y_add[offset : k_len + offset, :] = dictionary[idx]
        recomp+=y_add
        
        return recomp, signal - recomp


# In[ ]:


def calculate_residuals(signal,dictionary,decomposition):

    recomp = np.zeros(signal.shape)
    for i in range(decomposition.shape[0]):
        y_add = recomp.copy()
        offset = decomposition[i,1].astype(int)
        idx = decomposition[i,2].astype(int)
        k_len = dictionary[idx].shape[0]
        y_add[offset : k_len + offset, :] = dictionary[idx]
        recomp+=y_add
        
        return recomp, signal - recomp

