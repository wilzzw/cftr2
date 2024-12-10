import numpy as np

from math import ceil, floor

def central_der(i, yvals, dx):
    return (yvals[i+1] - yvals[i-1]) / (2 * dx)

def num_derivative(yvals, dx=1):
    padded_yvals = np.pad(yvals, pad_width=(1,1), constant_values=(2*yvals[0]-yvals[1], 2*yvals[-1]-yvals[-2]))

    def dydx(i):
        return central_der(i, padded_yvals, dx)
    dydx = np.vectorize(dydx)

    derivatives = dydx(np.arange(1, len(yvals)+1))
    return derivatives

# Angle between two vectors a and b
def vangle(a, b):
    assert a.shape == b.shape

    cosangle = np.sum(a*b, axis=-1) / (np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1))
    angle = np.arccos(cosangle)
    return np.degrees(angle)

def dihedral_angle(norm1, norm2):
    norm_angle = vangle(norm1, norm2)
    corrected_angle = np.where(norm_angle > np.pi/2, np.pi - norm_angle, norm_angle)
    return corrected_angle

# Angles are in radians
def R_x(a):
    a = np.atleast_1d(a)
    cos = np.cos(a)
    sin = np.sin(a)
    # Support array broadcasting
    return np.array([[np.ones(a.shape), np.zeros(a.shape), np.zeros(a.shape)], 
                     [np.zeros(a.shape), cos, -sin], 
                     [np.zeros(a.shape), sin, cos]]).transpose(2,0,1)
# Angles are in radians
def R_y(a):
    a = np.atleast_1d(a)
    cos = np.cos(a)
    sin = np.sin(a)
    # Support array broadcasting
    return np.array([[cos, np.zeros(a.shape), sin],
                     [np.zeros(a.shape), np.ones(a.shape), np.zeros(a.shape)],
                     [-sin, np.zeros(a.shape), cos]]).transpose(2,0,1)
# Angles are in radians
def R_z(a):
    a = np.atleast_1d(a)
    cos = np.cos(a)
    sin = np.sin(a)
    # Support array broadcasting
    return np.array([[cos, -sin, np.zeros(a.shape)],
                     [sin, cos, np.zeros(a.shape)],
                     [np.zeros(a.shape), np.zeros(a.shape), np.ones(a.shape)]]).transpose(2,0,1)


def always_array(items):
    return np.array([items]).flatten()

# Split input array a into equal shaped children arrays with overlapping array elements
def overlapping_split(a, num_element_childarray=2):
    # This will result in this number amount of child arrays:
    num_child_arrays = len(a) - num_element_childarray + 1
    output_child_set = []
    for i in range(num_child_arrays):
        if i+num_element_childarray >= len(a):
            output_child_set.append(a[i:])
        else:
            output_child_set.append(a[i:i+num_element_childarray])
    return output_child_set

def consecutive_split(a, diff=1):
    split_indices = np.argwhere(np.diff(a) > diff).flatten() + 1
    split_arrays = np.split(a, split_indices)
    return split_arrays

# Whether small_list is a sublist of big_list
# Easy with set operations
def is_sublist(small_list, big_list):
    for item in small_list:
        if item not in big_list:
            return False
    return True

def round_down(n, keep=1):
    multi10 = 10**(ceil(-np.log10(n)) + keep - 1)
    return floor(n*multi10) / multi10

# Data utilities
# Smoothing functions reduces noise in a vector of sequential data
# Smooth_loc() takes a position index and returns the smoothened value to replace
def smooth_loc(index, data_vector, window):
    if index - window < 0:
        select_for_smoothing = data_vector[:index+window+1]
    elif index + window + 1 >= data_vector.shape[0]:
        select_for_smoothing = data_vector[index-window:]
    else:
        select_for_smoothing = data_vector[index-window:index+window+1]
    return np.mean(select_for_smoothing)

# smoothing() takes a data_vector as argument and returns a vector of the same length but smoothened at all positions
def smoothing(data_vector, window=1):
    indices = np.arange(np.array(data_vector).shape[0])
    def specific_smooth(index):
        return smooth_loc(index, np.array(data_vector), window)
    smoothing_function = np.vectorize(specific_smooth)
    smoothed = smoothing_function(indices)
    return smoothed

# Returns selection string that would allow selection of a range of protein residues
# Currently only MDTraj
def select_resrange(resid_start, resid_end, prot_only=True, ca_only=True, analysis="MDTraj"):
    parent_str = "resSeq {start} to {end}".format(start=resid_start, end=resid_end)
    if prot_only:
        parent_str = "protein and {string}".format(string=parent_str)
    if ca_only:
        parent_str = "name CA and {string}".format(string=parent_str)
    return parent_str

def bootstrap_ci(original_sample, n_boot_samples, boot_sample_size=None, calculate_statistic=lambda a: np.mean(a), confidence_lvl=99):
    statistics = []
    if boot_sample_size is None:
        boot_sample_size = len(original_sample)
    
    for i in range(n_boot_samples):
        sample = np.random.choice(original_sample, size=boot_sample_size)
        stat = calculate_statistic(sample)
        statistics.append(stat)
    original_statistic = calculate_statistic(original_sample)
    interval = np.percentile(statistics, q=((100-confidence_lvl)/2, 100-(100-confidence_lvl)/2))
    return original_statistic, interval

def bootstrap_distrib_ci(original_sample, n_boot_samples, range_max, boot_sample_size=None, confidence_lvl=99):
    statistics = {o: [] for o in range(range_max+1)}
    if boot_sample_size is None:
        boot_sample_size = len(original_sample)
    
    for i in range(n_boot_samples):
        sample = np.random.choice(original_sample, size=boot_sample_size)
        for o in range(range_max+1):
            statistics[o].append(np.sum(sample == o)/len(sample))
    original_statistics = {o: np.mean(statistics[o]) for o in range(range_max+1)}
    intervals = {o: np.percentile(statistics[o], q=((100-confidence_lvl)/2, 100-(100-confidence_lvl)/2)) for o in range(range_max+1)}
    return original_statistics, intervals


# Used to modify file/directory names to reflect system information
def nmod(value, connect="_", insertion=""):
    if len(str(value)) > 0:
        return connect+insertion+str(value)
    return str(value)
