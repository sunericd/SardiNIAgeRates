
# coding: utf-8

# # Classes and Functions for the SardiNIA Aging Rates Project

# This is the main code notebook with the functions that are called upon by other notebooks for investigative purposes. It includes the "big bad" function that calculates biological ages (scroll down a bit).

# In[1]:

# Writes a Function.py script to be imported
ipython notebook --script


# In[ ]:

# Imports packages and functions (Pandas, NumPy, PyLab, SciPy, Scikit-Learn)
import pandas as pd
import numpy as np
import pylab as pl
import pickle
import os
import sys
import scipy
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.lda import LDA


# This is the class for organizing the main calculative functions and variables of each wave of data collection in the study for predicting physiological age.

# In[ ]:

class WaveInfo (object):
    class WaveData (object):
        def __init__ (self):
            raw_data         = None
            data_matrix      = None
            col_names        = None
            age_col_idx      = None
            id_col_idx       = None
            data_class_views = None
            class_vals       = None
            n_features       = None
        # A summary of the WaveData object
        def __str__ (self):
            s = "{} samples, {} features, {} classes\n".format(
                self.data_matrix.shape[0],self.data_matrix.shape[1],len(self.data_class_views))
            vals_str = ""
            counts_str = ""
            for i in range(len(self.data_class_views)):
                vals_str   += "{:6.1f}".format(self.class_vals[i])
                counts_str += "{:4d}  ".format(len(self.data_class_views[i]))
            s += vals_str   + "\n"
            s += counts_str + "\n"
            return (s)

    def __init__ (self, **kwargs):
        self.wavenum        = kwargs.get('wavenum'       , 0)
        self.filename_base  = kwargs.get('filename_base' , "")
        self.raw_data_fname = kwargs.get('raw_data_fname', "")
        self.num_train      = kwargs.get('num_train'     , 0)
        self.num_test       = kwargs.get('num_test'      , 0)
        self.age_start      = kwargs.get('age_start'     , 0)
        self.age_end        = kwargs.get('age_end'       , 100)
        self.bin_years      = kwargs.get('bin_years'     , 5)
        self.read_func      = kwargs.get('read_func'     , None)
        # This is a dict of ModelName : Best # of features
        self.best_models    = kwargs.get('best_models'   , {})
        # A cached read of the raw and cleaned-up data
        self.wave_data      = kwargs.get('wave_data'     , None)

    def raw_data_path (self):
        if len (self.raw_data_fname) > 1:
            return (os.path.join (Config.data_dir(), self.raw_data_fname))
        else:
            return (os.path.join (Config.data_dir(),
                Config.data_dist_base+'-Wave{}.{}'.format(
                self.wavenum,Config.wave_data_file_ext))
            )

    # These are defined later, but set right here
    # fake way of forward-declaration in python.
    def read_data_v1 (wavenum, what):
        print ('reading data from ',Config.wave_info(wavenum).raw_data_path())
        sys.stdout.flush()
        return (read_data_updated (Config.wave_info(wavenum).raw_data_path(), what))

    def read_data_v2 (wavenum):
        print ('reading data from ',Config.wave_info(wavenum).raw_data_path())
        sys.stdout.flush()
        return (read_w2 (Config.wave_info(wavenum).raw_data_path()))

    def read_data_v3 (wavenum, what):
        print ('reading data from ',Config.wave_info(wavenum).raw_data_path())
        sys.stdout.flush()
        return (read_data_updated2 (Config.wave_info(wavenum).raw_data_path(), what))


# #Configuration
# This includes various values/settings/etc determined empirically as the project progressed

# In[ ]:

class Config (object):
    data_dir_path = 'data'
    result_dir_path = 'results' # relative to data_dir_path
    graph_dir_path = 'Graphs' # relative to data_dir_path
# Directory for data files is returned by data_dir ():
#     return (os.path.join (Config.data_dir_path,Config.data_dist_base))
# The default raw data filename is set to:
#    data_dist_base+'-Wave{wavenum}.{wave_data_file_ext}
#    e.g. "2014-08-26-Sardinia-Wave1.tsv"
# The default raw data filename can be set using the raw_data_fname field, e.g.:
#    raw_data_fname = "newWave1.csv"

    data_dist_base = '2014-08-26-Sardinia'
#    data_dist_base = '2013-12-18-Sardinia-CleanUp-Data'
    mrmr_bin = os.path.join (os.path.expanduser("~"),'bin','mrmr')
    mrmr_sfx = '_mrmr_RS'   # change to +"_mrmr_10" for old splits, "_mrmr" for non-random-seed splits
    split_num_fmt = '_s{0:04d}'
    
    wave_data_file_ext = 'tsv'
    wave_data_file_delim = '\t'

    # N.B.: by default, wave_infos[wavenum].raw_data_fname = ""
    dist_wave_infos = {
        '2013-12-18-Sardinia-CleanUp-Data' : {
            1 : WaveInfo (
                wavenum        = 1,
                filename_base  = "sard_w1_split_120tr_13te",
                num_train      = 120,
                num_test       =  13,
                age_start      =  12,
                age_end        =  77,
                bin_years      =   5,
                read_func      = lambda: WaveInfo.read_data_v1 (1, 'features'),
                best_models    = {
                    'RandForClf':120,
                    'KNeighReg' :121,
                    'NuSVC'     :90, # not determined - made up!
                    'SVC'       :75, # not determined - made up!
                },
            ),
            2 : WaveInfo (
                wavenum        =  2,
                filename_base  = "sard_w2_split_96tr_10te",
                num_train      = 96,
                num_test       = 10,
                age_start      = 16,
                age_end        = 81,
                bin_years      =  5,
                read_func      = lambda: WaveInfo.read_data_v2 (2),
                best_models    = {
                    'RandForClf' : 95,
                    'KNeighReg'  : 91,
                    'NuSVC'      : 90, # not determined - made up!
                    'SVC'        : 75, # not determined - made up!
                },
            ),
            3 : WaveInfo (
                wavenum        =  3,
                filename_base  = "sard_w3_split_82tr_8te",
                num_train      = 82,
                num_test       =  8,
                age_start      = 20,
                age_end        = 80,
                bin_years      =  5,
                read_func      = lambda: WaveInfo.read_data_v1 (3, 'samples'),
                best_models    = {
                    'RandForClf': 84,
                    'KNeighReg' : 85,
                },
            ),
        },
        '2014-08-26-Sardinia' : {
            1 : WaveInfo (
                wavenum        = 1,
                filename_base  = "sard_w1_split_119tr_13te",
                num_train      = 119,
                num_test       =  13,
                age_start      =  15,
                age_end        =  80,
                bin_years      =   5,
                read_func      = lambda: WaveInfo.read_data_v3 (1, 'samples'),
                best_models    = {
                    'RandForClf':120,
                    'KNeighReg' :121,
                    'NuSVC'     :90, # not determined - made up!
                    'SVC'       :75, # not determined - made up!
                },
            ),
            2 : WaveInfo (
                wavenum        =  2,
                filename_base  = "sard_w2_split_161tr_18te",
                num_train      = 161,
                num_test       = 18,
                age_start      = 15,
                age_end        = 80,
                bin_years      =  5,
                read_func      = lambda: WaveInfo.read_data_v3 (2, 'samples'),
                best_models    = {
                    'RandForClf' : 95,
                    'KNeighReg'  : 91,
                    'NuSVC'      : 90, # not determined - made up!
                    'SVC'        : 75, # not determined - made up!
                },
            ),
            3 : WaveInfo (
                wavenum        =   3,
                filename_base  = "sard_w3_split_103tr_12te",
                num_train      = 103,
                num_test       =  12,
                age_start      =  18,
                age_end        =  83,
                bin_years      =   5,
                read_func      = lambda: WaveInfo.read_data_v3 (3, 'samples'),
                best_models    = {
                    'RandForClf': 95, # made up, based on wave 2!
                    'KNeighReg' : 91, # made up, based on wave 2!
                },
            ),
        },
    }

    wave_infos = dict (dist_wave_infos [data_dist_base])

    def set_data_dist_base (data_dist_base):
        if data_dist_base in ['2014-08-26-Sardinia','2014-08-26']:
            Config.data_dist_base = '2014-08-26-Sardinia'
        elif data_dist_base in ['2013-12-18-Sardinia-CleanUp-Data','2013-12-18-Sardinia','2013-12-18']:
            Config.data_dist_base = '2013-12-18-Sardinia-CleanUp-Data'
        print ('Setting Config.data_dist_base to "{}"'.format (Config.data_dist_base))
        Config.wave_infos = dict (Config.dist_wave_infos [Config.data_dist_base])

    def data_dir ():
       return (os.path.join (Config.data_dir_path,Config.data_dist_base))
    
    def result_dir ():
       return (os.path.join (Config.data_dir(), Config.result_dir_path))
    
    def graph_dir ():
       return (os.path.join (Config.data_dir(), Config.graph_dir_path))

    
    def wave_info (wavenum):
        Split.filename_base = Config.wave_infos[wavenum].filename_base
        return (Config.wave_infos[wavenum])

    def wave_data (wavenum):
        wi = Config.wave_info (wavenum)
        if wi.wave_data is None:
            wd = WaveInfo.WaveData()
            wd.raw_data = wi.read_func()
            ages = wd.raw_data.Age.values
            wd.data_matrix, wd.col_names, wd.age_col_idx, wd.id_col_idx = clean_convert(wd.raw_data)
            wd.data_class_views, wd.class_vals = bin_data (wi.age_start, wi.age_end, wi.bin_years, wd.data_matrix, ages)
            wd.n_features = len (wd.col_names) - 2
            # cache the result
            wi.wave_data = wd
        return (wi.wave_data)


# #Main Class for holding Split information

# This is the class called upon to process the SardiNIA dataset, predict biological/physiological ages, and store the values.

# In[ ]:

class Split (object):
    filename_base = Config.wave_infos[1].filename_base
    filename_base_pro2 = filename_base+"_pro2"
    def __init__ (self):
        # These are 2D numpy arrays, features in columns, samples in rows
        self.train_set = None
        self.test_set = None
        self.n_classes = None

        # classed_labels are center values for each bin
        # labels are the continuously varying ages (unbinned) 
        self.test_labels = []
        self.train_labels = []
        self.test_classed_labels = []
        self.train_classed_labels = []
        # cached dict of class values
        self.class_vals = None
        # Participant IDs
        self.train_id = []
        self.test_id = []
        self.rand_seed = None
        
        # N.B.: Anything added here must be added to copy()!

#    def copy (self,from_sp):
#        self.train = from_sp.train
#        self.test = from_sp.test
#        self.test_labels = from_sp.test_labels
#        self.test_classed_labels = from_sp.test_classed_labels
#        self.train_labels = from_sp.train_labels
#        self.train_classed_labels = from_sp.train_classed_labels

#        self.train_vstack = from_sp.train_vstack
#        self.test_vstack = from_sp.test_vstack
#        self.train_3d = from_sp.train_3d
#        self.test_3d = from_sp.test_3d

#        self.sorted_train = from_sp.sorted_train
#        self.sorted_test = from_sp.sorted_test
#        self.stand_train = from_sp.stand_train
#        self.stand_test = from_sp.stand_test
#        self.weigh_train = from_sp.weigh_train
#        self.weigh_test = from_sp.weigh_test
#        return (self)
    
    def copy (self,from_sp):
        # This will still not make actual copies of numpys burried in lists!!!
        # They will be references to the same original numpys!
        for k in from_sp.__dict__.keys():
            if (type(from_sp.__dict__[k]) == np.ndarray):
                self.__dict__[k] = from_sp.__dict__[k].copy()
            else:
                self.__dict__[k] = from_sp.__dict__[k]
        return (self)

    def get_fname (self, splitnum, protocol = 3):
        if protocol == 3:
            fname = Split.filename_base+Config.split_num_fmt.format(splitnum)+".pickle"
            return (os.path.join(Config.data_dir(),Split.filename_base,fname))
        else:
            fname_pro2 = Split.filename_base_pro2+Config.split_num_fmt.format(splitnum)+".pickle"
            return (os.path.join(Config.data_dir(),Split.filename_base_pro2,fname_pro2))
        
    def save (self, filename, protocol=3):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol)

    def read (self, filename):
        if sys.version_info >= (3, 0):
            split = self.read_py3_pickle(filename)
        else:
            split = self.read_py2_from_py3_pickle(filename)

        # This is to read older versions of Split
        if split.test_set is None and not split.test is None:
            split.test_set = np.vstack(split.test)
        if split.train_set is None and not split.train is None:
            split.train_set = np.vstack(split.train)
        return (self)

    def read_py3_pickle (self, filename):
        rd = None
        with open(filename, 'rb') as f:
            rd = pickle.load(f)
        return (self.copy (rd))
    
    def read_py2_from_py3_pickle (self, filename):
        rd = None
        with open(filename, 'rb') as f:
            rd = load_py2_from_py3_pickle(f)
        return (self.copy (rd))

    def train_test(self, train_amount, test_amount, views, class_vals, age_col_idx, id_col_idx, rand_seed = None):
        train = []
        test = []
        train_labels = []
        train_classed_labels = []
        test_labels = []
        test_classed_labels = []
        train_id = []
        test_id = []
        index = 0
        for view in views:
            bv = class_vals[index]
            index = index + 1 

            new_view = np.insert(view, 0, bv, axis=1)
            # This insert makes age_col_idx and id_col_idx increase by 1.
            new_age_col_idx = age_col_idx+1
            new_id_col_idx = id_col_idx+1
            if rand_seed is not None:
                random = np.random.RandomState(rand_seed).permutation(new_view)
                self.rand_seed = rand_seed
            else:
                random = np.random.permutation(new_view)
                self.rand_seed = None
            train_classed_labels.append(random[:train_amount,0])
            test_classed_labels.append(random[train_amount:train_amount+test_amount:,0])
            train_labels.append(random[:train_amount, new_age_col_idx])
            test_labels.append(random[train_amount:train_amount+test_amount:, new_age_col_idx])
            train_id.append(random[:train_amount, new_id_col_idx])
            test_id.append(random[train_amount:train_amount+test_amount:,new_id_col_idx])

            class_data = np.delete(random, new_age_col_idx , axis=1)
            class_data = np.delete(class_data, new_id_col_idx , axis=1)
            class_data = np.delete(class_data, 0 , axis=1)

            train.append(class_data[:train_amount, :])
            test.append(class_data[train_amount:train_amount+test_amount, :]) 
        
        self.train_labels = np.concatenate(train_labels)
        self.train_classed_labels = np.concatenate(train_classed_labels)
        self.test_labels = np.concatenate(test_labels)
        self.test_classed_labels = np.concatenate(test_classed_labels)
        self.train_id = np.concatenate(train_id)
        self.test_id = np.concatenate(test_id)

        self.train_set = np.vstack(train)
        self.test_set = np.vstack(test)
        return (self)

    def load_wave_train_test_RS (self, wavenum, split_num):
        wi = Config.wave_info (wavenum)
        wd = Config.wave_data (wavenum)
        self.train_test(wi.num_train, wi.num_test,
            wd.data_class_views, wd.class_vals, wd.age_col_idx, wd.id_col_idx,
            split_num)

    def get_class_vals(self):
        if self.class_vals is None:
            class_vals_dict = {}
            for class_label in self.train_classed_labels:
                class_vals_dict[class_label] = None
            self.class_vals = sorted(class_vals_dict.keys())
            self.n_classes = len (self.class_vals)
        return (self.class_vals)

    def get_n_classes(self):
        if self.n_classes is None:
            self.get_class_vals()
        return (self.n_classes)

    def get_train_3d (self):
        if self.train_3d is None:
            self.train_3d = get_class_mat_list (self.train_set, self.train_classed_labels)
        return (self.train_3d)

    def get_test_3d (self):
        if self.test_3d is None:
            self.test_3d = get_class_mat_list (self.test_set, self.test_classed_labels)
        return (self.test_3d)

    def sort_stand_weigh(self, feature_weights):
        i = np.argsort(feature_weights)
        sorted_train = self.get_train_vstack()[:,i]
        self.sorted_train = np.fliplr(sorted_train)
        sorted_test = self.get_test_vstack()[:,i]
        self.sorted_test = np.fliplr(sorted_test)
        self.stand(self.sorted_train, self.sorted_test)
        self.weigh_train = np.multiply(self.stand_train, feature_weights)
        self.weigh_test = np.multiply(self.stand_test, feature_weights)
        return (self)

    def norm_weigh_sort(self, feature_weights):
        self.normalize (self.train_set, self.test_set)
        self.apply_weights (feature_weights)
        sorted_weights = self.sort_by_weight (feature_weights)
        return (sorted_weights)

    def stand_sort(self, feature_weights):
        self.stand (self.train_set, self.test_set)
        sorted_weights = self.sort_by_weight (feature_weights)
        return (sorted_weights)

    def stand_weigh_sort(self, feature_weights):
        self.stand (self.train_set, self.test_set)
        self.apply_weights (feature_weights)
        sorted_weights = self.sort_by_weight (feature_weights)
        return (sorted_weights)

    def get_trimmed_features (self, num_feat):
        new_train = self.train_set[:,:num_feat]
        new_test = self.test_set[:,:num_feat]
        return (new_train, new_test)

    def stand (self, train, test):
        scaler = StandardScaler()
        self.train_set = scaler.fit_transform(train)
        self.test_set = scaler.transform(test)
        return (self)
    
    def normalize (self, train, test):
        self.train_set = train.copy() 
        mins, maxs = normalize_by_columns (self.train_set)
        self.test_set = test.copy() 
        normalize_by_columns (self.test_set, mins, maxs)
        return (self)

    def sort_by_weight (self, feature_weights):
        i = np.argsort(feature_weights)
        self.train_set = self.train_set[:,i]
        self.train_set = np.fliplr(self.train_set)
        self.test_set = self.test_set[:,i]
        self.test_set = np.fliplr(self.test_set)
        feature_weights = np.sort(feature_weights)
        feature_weights[:] = feature_weights[::-1]
        return (feature_weights)

    def apply_weights (self, feature_weights):
        self.train_set = np.multiply (self.train_set, feature_weights)
        self.test_set = np.multiply (self.test_set, feature_weights)
        return (self)



# In[ ]:

def get_class_mat_list (mat, class_labels):
    assert (len(mat) == len(class_labels))
    class_label_dict = {}
    class_mats = []
    class_label_idx = 0
    for samp_idx in range (len (mat)):
        class_label = class_labels[samp_idx]
        if not class_label in class_label_dict:
            class_label_dict[class_label] = class_label_idx
            class_mats.append (mat[samp_idx])
            class_label_idx += 1
        else:
            class_idx = class_label_dict[class_label]
            class_mats[class_idx] = np.vstack ([class_mats[class_idx],mat[samp_idx]])
    return (np.array(class_mats))
    


# Below are several read_data functions that allow for optimized cleaning for individual waves of data. Some are older prototypes.

# In[ ]:

def read_data(file_name):
    x = pd.read_csv(file_name,sep=Config.wave_data_file_delim, na_values=[])
    x = x.dropna(axis=1, how='all')
    x = x.dropna(how='all')
    y, z = x.shape
    x = x.dropna(axis=1, thresh=(y/3))
    y, z = x.shape
    x = x.dropna(thresh=(z/2))
    y, z = x.shape
    x = x.dropna(axis=1, thresh=(y*2/3))
    y, z = x.shape
    x = x.dropna(thresh=(z*3/4))
    y, z = x.shape
    x = x.dropna(axis=1, thresh=(y*9/10))
    y, z = x.shape
    x = x.dropna(thresh=(z*4/5))
    y, z = x.shape
    x = x.dropna(axis=1, thresh=(y*9/10))
    y, z = x.shape
    x = x.dropna(how='any')
    return x


# In[ ]:

def read_data_updated(file_name, what_to_prioritize):
    x = pd.read_csv(file_name,sep=Config.wave_data_file_delim, na_values=[])
    x = x.dropna(axis=1, how='all')
    x = x.dropna(how='all')
    x = x[pd.notnull(x['pwv'])]
    x = x[pd.notnull(x['exmBMI'])]
    if 'exmBPsys_jbs' in x:
        x = x[pd.notnull(x['exmBPsys_jbs'])]
    else:
        x = x[pd.notnull(x['exmBPsys'])]
    x = x[pd.notnull(x['exmWaist'])]
    x = x[pd.notnull(x['labsColesterolo'])]
    threshold_num = 0.05
    if what_to_prioritize is 'features':
        for i in range (47):
            x = x.dropna(thresh=(x.shape[1]*threshold_num))
            x = x.dropna(axis=1, thresh=(x.shape[0]*threshold_num))
            threshold_num += 0.02
        x = x.dropna(how='any')
    elif what_to_prioritize is 'samples':
        for i in range (47):
            x = x.dropna(axis=1, thresh=(x.shape[0]*threshold_num))
            x = x.dropna(thresh=(x.shape[1]*threshold_num))
            threshold_num += 0.02
        x = x.dropna(axis=1, how='any')
    else:
        return("Need to specify what to prioritize: 'features' or 'samples'")
    return (x)


# In[ ]:

def read_data_updated2(file_name, what_to_prioritize):
    x = pd.read_csv(file_name,sep=Config.wave_data_file_delim, na_values=[])
    x = x.dropna(axis=1, how='all')
    x = x.dropna(how='all')
    # We're not singling out "must keep" columns like before:
    #   pwv, exmBMI, exmBPsys_jbs, exmBPsys, exmWaist, labsColesterolo
    # Instead we're dropping unwanted columns first.
    x = clean_data (x)

    threshold_num = 0.05
    if what_to_prioritize is 'features':
        for i in range (47):
            x = x.dropna(thresh=(x.shape[1]*threshold_num))
            x = x.dropna(axis=1, thresh=(x.shape[0]*threshold_num))
            threshold_num += 0.02
        x = x.dropna(how='any')
    elif what_to_prioritize is 'samples':
        for i in range (47):
            x = x.dropna(axis=1, thresh=(x.shape[0]*threshold_num))
            x = x.dropna(thresh=(x.shape[1]*threshold_num))
            threshold_num += 0.02
        x = x.dropna(axis=1, how='any')
    else:
        return("Need to specify what to prioritize: 'features' or 'samples'")
    return (x)


# In[ ]:

# A specially developed trial for Wave 2 data, which had certain characteristics (missing and added features) different.
def read_w2 (file_name):
    x = pd.read_csv(file_name,sep=Config.wave_data_file_delim, na_values=[])
    x = x.dropna(axis=1, how='all')
    x = x.dropna(how='all')
    x = x[pd.notnull(x['exmBMI'])]
    if 'exmBPsys_jbs' in x:
        x = x[pd.notnull(x['exmBPsys_jbs'])]
    else:
        x = x[pd.notnull(x['exmBPsys'])]
    x = x[pd.notnull(x['exmWaist'])]
    x = x[pd.notnull(x['labsColesterolo'])]
    x = x[pd.notnull(x['vasPSV'])]
    x = x[pd.notnull(x['vasIMT'])]
    x = x[pd.notnull(x['vasvti'])]
    y, z = x.shape
    x = x.dropna(axis=1, thresh=(y/3))
    y, z = x.shape
    x = x.dropna(thresh=(z/2))
    y, z = x.shape
    x = x.dropna(axis=1, thresh=(y/2))
    y, z = x.shape
    x = x.dropna(thresh=(z*4/5))
    y, z = x.shape
    x = x.dropna(axis=1, thresh=(y*9/10))
    y, z = x.shape
    x = x.dropna(thresh=(z*9/10))
    y, z = x.shape
    x = x.dropna(axis=1, thresh=(y*99/100))
    y, z = x.shape
    x = x.dropna(how='any')
    return (x)


# In[ ]:

# Unfinished harmonized feature function
#def read_data_updated_harmonized(file_name, what_to_prioritize):
#    x = pd.read_csv(file_name, na_values=[])
#    x = x.dropna(axis=1, how='all')
#    x = x.dropna(how='all')
#    x = x[pd.notnull(x['pwv'])]
#    x = x[pd.notnull(x['exmBMI'])]
#    x = x[pd.notnull(x['exmBPsys_jbs'])]
#    x = x[pd.notnull(x['exmWaist'])]
 #   x = x[pd.notnull(x['labsColesterolo'])]
  #  x = x[pd.notnull(x['vasPSV'])]
   # x = x[pd.notnull(x['vasIMT'])]
#    x = x[pd.notnull(x['vasvti'])]
 #   threshold_num = 0.05
  #  if what_to_prioritize is 'features':
   #     for i in range (47):
#            x = x.dropna(thresh=(x.shape[1]*threshold_num))
 #           x = x.dropna(axis=1, thresh=(x.shape[0]*threshold_num))
#         threshold_num += 0.02
 #       x = x.dropna(how='any')
  #  elif what_to_prioritize is 'samples':
   #     for i in range (47):
    #        x = x.dropna(axis=1, thresh=(x.shape[0]*threshold_num))
     #       x = x.dropna(thresh=(x.shape[1]*threshold_num))
      #      threshold_num += 0.02
       # x = x.dropna(axis=1, how='any')
#    else:
 #       return("Need to specify what to prioritize: 'features' or 'samples'")
  #  return (x)


# In[ ]:

#def read_w2_harmonized (file_name):
 #   x = pd.read_csv(file_name, na_values=[])
  #  x = x.dropna(axis=1, how='all')
   # x = x.dropna(how='all')
    #x = x[pd.notnull(x['pwv'])]
#    x = x[pd.notnull(x['exmBMI'])]
 #   x = x[pd.notnull(x['exmBPsys_jbs'])]
  #  x = x[pd.notnull(x['exmWaist'])]
   # x = x[pd.notnull(x['labsColesterolo'])]
    #x = x[pd.notnull(x['vasPSV'])]
#    x = x[pd.notnull(x['vasIMT'])]
 #   x = x[pd.notnull(x['vasvti'])]
  #  y, z = x.shape
   # x = x.dropna(axis=1, thresh=(y/3))
    #y, z = x.shape
#    x = x.dropna(thresh=(z/2))
 #   y, z = x.shape
  #  x = x.dropna(axis=1, thresh=(y/2))
   # y, z = x.shape
    #x = x.dropna(thresh=(z*4/5))
#    y, z = x.shape
 #   x = x.dropna(axis=1, thresh=(y*9/10))
  #  y, z = x.shape
   # x = x.dropna(thresh=(z*9/10))
    #y, z = x.shape
#    x = x.dropna(axis=1, thresh=(y*99/100))
 #   y, z = x.shape
  #  x = x.dropna(how='any')
   # return (x)


# Here are some data cleaning and conversion functions.

# In[ ]:

# Drops columns with data that is not pertinent to our physiological prediction of age
# or is likely to add unwanted nonphysiological influences to the prediction.
def clean_data(df):
    drop_cols = [
        'id_sir', 'id_mad', 'pwvDate', 'FirstVisitDate', 'SecondVisitDate', 'ThirdVisitDate',
        'FourthVisitDate', 'Birthdate', 'Scandate', 'date_neo', 'Visit', 'Wave', 'Subject#',
        'Subject_ID[4%]', 'Subject_ID[38%]', 'Subject_ID[66%]', 'Occupation', 'Education',
        'MaritalStatus',
        'SpirometerPrELA','SpirometerELA',
    ]
    
    for col in drop_cols:
        try:
            df = df.drop(col,1)
        except:
            pass
    return (df)


# In[ ]:

# Converts dataset into data, names, and idx's for the Age and ID cols.
def convert_data(df):
    data = df.as_matrix()
    col_names = df.columns.values
    return (data, col_names, df.columns.get_loc("Age"), df.columns.get_loc("id_individual"))


# In[ ]:

# Combined function -- called on in workflow
def clean_convert(df):
    return (convert_data(clean_data(df)))


# In[ ]:

# This bin_data relies/assumes that data_matrix is sorted by age!
def bin_data_OLD(start, end, size, data_matrix, age_col):
    bins = range (start, end+size, size)
    counts,bins = np.histogram(age_col,bins=bins)
    views=[]
    index=0
    for i in range (0,len(counts)):
        views.append (data_matrix[index : index+counts[i],:])
        index = index+counts[i]
    views.pop()
    # initialize the class_values array (center of each bin)
    class_vals = [x+(size/2.0) for x in range(start, end, size)]
    class_vals.pop()
    return (views, class_vals)


# In[ ]:

def bin_data(start, end, size, data_matrix, age_col):
    bins = range (start, end+size, size)
    digitized = np.digitize(age_col, bins)

    views=[]
    for i in range (1,len(bins)):
        views.append (data_matrix[digitized == i,:])
#    views.pop()
    # initialize the class_values array (center of each bin)
    class_vals = [x+(size/2.0) for x in range(start, end, size)]
#    class_vals.pop()
    return (views, class_vals)


# In[ ]:

def normalize_by_columns ( full_stack, mins = None, maxs = None ):
    """This is a global function to normalize a matrix by columns.
    If numpy 1D arrays of mins and maxs are provided, the matrix will be normalized against these ranges
    Otherwise, the mins and maxs will be determined from the matrix, and the matrix will be normalized
    against itself. The mins and maxs will be returned as a tuple.
    Out of range matrix values will be clipped to min and max (including +/- INF)
    zero-range columns will be set to 0.
    NANs in the columns will be set to 0.
    The normalized output range is hard-coded to 0-100
    """
    # Edge cases to deal with:
    # Range determination:
    # 1. features that are nan, inf, -inf
    # max and min determination must ignore invalid numbers
    # nan -> 0, inf -> max, -inf -> min
    # Normalization:
    # 2. feature values outside of range
    # values clipped to range (-inf to min -> min, max to inf -> max) - leaves nan as nan
    # 3. feature ranges that are 0 result in nan feature values
    # 4. all nan feature values set to 0

    # Turn off numpy warnings, since we're taking care of invalid values explicitly
    oldsettings = np.seterr(all='ignore')
    if (mins is None or maxs is None):
        # mask out NANs and +/-INFs to compute min/max
        full_stack_m = np.ma.masked_invalid (full_stack, copy=False)
        maxs = full_stack_m.max (axis=0)
        mins = full_stack_m.min (axis=0)

    # clip the values to the min-max range (NANs are left, but +/- INFs are taken care of)
    full_stack.clip (mins, maxs, full_stack)
    # remake a mask to account for NANs and divide-by-zero from max == min
    full_stack_m = np.ma.masked_invalid (full_stack, copy=False)

    # Normalize
    full_stack_m -= mins
    full_stack_m /= (maxs - mins)
    # Left over NANs and divide-by-zero from max == min become 0
    # Note the deep copy to change the numpy parameter in-place.
    full_stack[:] = full_stack_m.filled (0) * 100.0

    # return settings to original
    np.seterr(**oldsettings)

    return (mins,maxs)



# # Feature Selection

# ### Fisher

# In[ ]:

def Fisher(split):
    """Takes a FeatureSet_Discrete as input and calculates a Fisher score for
    each feature. Returns a newly instantiated instance of FisherFeatureWeights.

    For:
    N = number of classes
    F = number of features
    It = total number of images in training set
    Ic = number of images in a given class
    """

    if split == None:
        import inspect
        form_str = 'You passed in a None as a training set to the function {0}.{1}'	
        raise ValueError( form_str.format( cls.__name__, inspect.stack()[1][3] ) )

    # we deal with NANs/INFs separately, so turn off numpy warnings about invalid floats.
    oldsettings = np.seterr(all='ignore')

    def get_train_3d (self):
        if self.train_3d is None:
            self.train_3d = get_class_mat_list (self.train_set, self.train_classed_labels)
        return (self.train_3d)

    def get_test_3d (self):
        if self.test_3d is None:
            self.test_3d = get_class_mat_list (self.test_set, self.test_classed_labels)
        return (self.test_3d)

    #class_mats = split.get_train_3d()
    class_mats = get_class_mat_list (split.train_set, split.train_classed_labels)
    # 1D matrix 1 * F
    population_means = np.mean( split.train_set, axis = 0 )
    n_classes = class_mats.shape[0]
    n_features = split.train_set.shape[1]

    # 2D matrix shape N * F
    intra_class_means = np.empty( [n_classes, n_features] )
    # 2D matrix shape N * F
    intra_class_variances = np.empty( [n_classes, n_features] )

    class_index = 0
    for class_feature_matrix in class_mats:
        intra_class_means[ class_index ] = np.mean( class_feature_matrix, axis=0 )
    # Note that by default, numpy divides by N instead of the more common N-1, hence ddof=1.
        intra_class_variances[ class_index ] = np.var( class_feature_matrix, axis=0, ddof=1 )
        class_index += 1

    # 1D matrix 1 * F
    # we deal with NANs/INFs separately, so turn off numpy warnings about invalid floats.
    # for the record, in numpy:
    # 1./0. = inf, 0./inf = 0., 1./inf = 0. inf/0. = inf, inf/inf = nan
    # 0./0. = nan, nan/0. = nan, 0/nan = nan, nan/nan = nan, nan/inf = nan, inf/nan = nan
    # We can't deal with NANs only, must also deal with pos/neg infs
    # The masked array allows for dealing with "invalid" floats, which includes nan and +/-inf
    denom = np.mean( intra_class_variances, axis = 0 )
    denom[denom == 0] = np.nan
    feature_weights_m = np.ma.masked_invalid (
            ( np.square( population_means - intra_class_means ).sum( axis = 0 ) /
        (n_classes - 1) ) / denom
        )
    # return numpy error settings to original
    np.seterr(**oldsettings)

    # the filled(0) method of the masked array sets all nan and infs to 0
    fisher_values = feature_weights_m.filled(0).tolist()

    return (fisher_values)


# ### Pearson

# In[ ]:

def Pearson(split):
    """Calculate regression parameters and correlation statistics that fully define
    a continuous classifier.

    At present the feature weights are proportional the Pearson correlation coefficient
    for each given feature."""

    from scipy import stats

    # Known issue: running stats.linregress() with np.seterr (all='raise') has caused
    # arithmetic underflow (FloatingPointError: 'underflow encountered in stdtr' )
    # I think this is something we can safely ignore in this function, and return settings
    # back to normal at the end. -CEC
    np.seterr (under='ignore')

    matrix = split.train_set
    #FIXME: maybe add some dummyproofing to constrain incoming array size

    #r_val_sum = 0
    r_val_squared_sum = 0
    #r_val_cubed_sum = 0
    
    ages = split.train_labels

    ground_truths = np.array( [float(val) for val in ages] )
    pearson_coeffs = np.zeros(matrix.shape[1])

    for feature_index in range( matrix.shape[1] ):
        slope, intercept, pearson_coeff, p_value, std_err = stats.linregress(
            ground_truths, matrix[:,feature_index]
        )

        pearson_coeffs[feature_index] = pearson_coeff
        r_val_squared_sum += pearson_coeff * pearson_coeff

# We're just returning the pearsons^2 now...
#    pearson_values = [val*val / r_val_squared_sum for val in pearson_coeffs ]
#    pearson_coeffs = (pearson_coeffs * pearson_coeffs) / r_val_squared_sum
    pearson_coeffs *= pearson_coeffs
    

    # Reset numpy
    np.seterr (all='raise')

    return pearson_coeffs


# ### PCA

# In[ ]:

def pca(split):
    from sklearn.decomposition import PCA
    pca = PCA()
    new_train = pca.fit_transform(split.get_train_vstack())
    new_test = pca.transform(split.get_test_vstack())
    return(new_train, new_test)


# ### LDA

# In[ ]:

def lda(train, test, split):
    lda = LDA()
    lda_train = lda.fit(train, split.train_classed_labels).transform(train)
    lda_test = lda.transform(test)
    return (lda_train, lda_test)


# ### mRMR

# In[ ]:

def mrmr(split, **kwargs):
    import subprocess
    import tempfile
    import os
    import time
    
    if 'thresh' in kwargs:
        thresh=kwargs['thresh']
    else:
        thresh=0.1
    if 'sigfigs' in kwargs:
        sigfigs=kwargs['sigfigs']
    else:
        sigfigs=7

    weights = []  
    class_labels = split.train_classed_labels.reshape(len(split.train_classed_labels),1)
    names = [float(i) for i in range (0,(split.train_set).shape[1]+1)]
    data = np.append(class_labels, split.train_set, axis=1)
    data = [names, data]
    data = np.vstack(data)
    tmpfile = tempfile.NamedTemporaryFile(delete=False).name
    # only save 5 sig-figs in text file
    np.savetxt(tmpfile, data, fmt='%.{}g'.format(sigfigs), delimiter=",")

    ignore_lines = True
    cmd_list = [Config.mrmr_bin, "-i",tmpfile, "-n", str(split.train_set.shape[1]),
                                "-s", str(split.train_set.shape[0]), '-t', str(thresh)]
    cmd_str = subprocess.Popen(cmd_list, stdout=subprocess.PIPE).stdout
#    cmd_str = subprocess.Popen([Config.mrmr_bin, "-i","/home/suned/mrmr/test_lung_s3.csv"], stdout=subprocess.PIPE).stdout
#    t0 = time.time()
    for line in cmd_str:
#        print ("time: "+str(time.time() - t0))
#        t0 = time.time()
        line = line.decode("utf-8").strip()
#        print (line)
        # Ignore everything until lines like this:
        #    *** mRMR features *** 
        # Order 	 Fea 	 Name 	 Score
        if '*** mRMR features ***' in line:
            ignore_lines = False
            continue
        if ignore_lines:
                continue
        if 'Order' in line:
                continue
        cols = line.split()
        if (len(cols) != 4):
                continue
        try:
            cols = [int(cols[0]), int(cols[1]), float(cols[3])]
        except:
            continue
        weights.append(cols)
    if len (weights) > 0:
        weights = np.vstack(weights)
        weights = weights[weights[:,1].argsort()]
        weights = weights[:,2]
        os.unlink (tmpfile)
    else:
        raise ValueError (" ".join(cmd_list)+"\mrmr returned no weights")
    return weights


# In[ ]:

def read_mrmr(wavenum, splitnum):
    mrmr_filename_base = Config.wave_info(wavenum).filename_base
    mrmr_filename_base += Config.mrmr_sfx
    mrmr_file_name = mrmr_filename_base+Config.split_num_fmt.format(splitnum)+".pickle"
    mrmr_file_path = os.path.join(Config.data_dir(),mrmr_filename_base,mrmr_file_name)
    with open(mrmr_file_path, 'rb') as f:
        mrmr_weights = pickle.load(f)
    return (mrmr_weights)


# ### Random Weights

# In[7]:

def random_weights(split, **kwargs):
    ''' The strategery here is to assign random weights to the features
    '''
    n_features = split.train_set.shape[1]
    if 'rand_seed' in kwargs:
        weights = np.random.RandomState(kwargs['rand_seed']).rand(n_features)
    else:
        weights = np.random.rand(n_features)
    return weights


# ## Classification

# In[ ]:

# map used to get model info by name.
# Names are populated where the functions are defined.
ModelInfoByModelName = {}


# ### WND

# In[ ]:

def marg_prob_to_pred_value (marg_probs, class_vals):
    weighted = np.array(marg_probs)*np.array(class_vals)
    return (np.sum(weighted))


# In[ ]:

def WND5(train_classed_data, test_classed_data, classnames_list, split):
    """
    Don't call this function directly, use the wrapper functions
    DiscreteBatchClassificationResult.New() (for test sets) or
    DiscreteImageClassificationResult.NewWND5() (for single images)
    Both of these functions have dummyproofing.

    If you're using this function, your training set data is not continuous
    for N images and M features:
    trainingset is list of length L of N x M numpy matrices
    testtile is a 1 x M list of feature values
    NOTE: the trainingset and test image must have the same number of features!!!
    AND: the features must be in the same order!!
    Returns an instance of the class DiscreteImageClassificationResult
    FIXME: what about tiling??
    """
    n_test_samples = test_classed_data.shape[0]
    n_train_samples = train_classed_data.shape[0]
    predicted_classes = np.zeros(n_test_samples)
    predicted_values = np.zeros(n_test_samples)
    
    epsilon = np.finfo( np.float ).eps
    testimg_idx = 0
    trainimg_idx = 0
    
    for testimg_idx in range( n_test_samples ):
        test_class_label = split.test_classed_labels[testimg_idx]

        # initialize
        class_dists = {}
        class_counts = {}

        for trainimg_idx in range( n_train_samples ):
            train_class_label = split.train_classed_labels[trainimg_idx]
            if not train_class_label in class_dists:
                class_dists [train_class_label] = 0.0
                class_counts[train_class_label] = 0.0

            dists = np.absolute (train_classed_data [trainimg_idx] - test_classed_data [testimg_idx])
            w_dist = np.sum( dists )
            if w_dist > epsilon:
                class_counts[train_class_label] += 1.0
            else:
                continue

            w_dist = np.sum( np.square( dists ) )
            # The exponent -5 is the "5" in "WND5"
            class_dists[ train_class_label ] += w_dist ** -5

        
        class_idx = 0
        class_similarities = [0]*len(class_dists)
        for class_label in classnames_list:
            class_similarities[class_idx] = class_dists[class_label] / class_counts[class_label]
            class_idx += 1

        norm_factor = sum( class_similarities )
        marg_probs = np.array( [ x / norm_factor for x in class_similarities ] )

        predicted_class_idx = marg_probs.argmax()

        predicted_classes[testimg_idx] = classnames_list[ predicted_class_idx ]
        predicted_values[testimg_idx] = marg_prob_to_pred_value (marg_probs, classnames_list)

    return (predicted_classes, predicted_values, split.test_labels)

ModelInfoByModelName['WND5'] = [WND5,'C']


# ### SVC

# In[ ]:

def svc(train_classed_data, test_classed_data, classnames_list, split):
    from sklearn.svm import SVC
    clf = SVC(kernel='linear', probability=True)
    clf.fit(train_classed_data, split.train_classed_labels)
    predicted_classes = clf.predict(test_classed_data)
    predicted_values = np.zeros(test_classed_data.shape[0])
    testimg_idx = 0
    for sample in test_classed_data:
        predicted_values[testimg_idx] = marg_prob_to_pred_value (clf.predict_proba(sample), classnames_list)
        testimg_idx += 1
    actual = split.test_labels
    return (predicted_classes, predicted_values, actual)

ModelInfoByModelName['SVC'] = [svc,'C']


# ### NuSVC

# In[ ]:

def nu_svc(train_classed_data, test_classed_data, classnames_list, split):
    from sklearn.svm import NuSVC
    clf = NuSVC(nu = 0.3, kernel='linear', probability=True)
    clf.fit(train_classed_data, split.train_classed_labels)
    predicted_classes = clf.predict(test_classed_data)
    predicted_values = np.zeros(test_classed_data.shape[0])
    testimg_idx = 0
    for sample in test_classed_data:
        predicted_values[testimg_idx] = marg_prob_to_pred_value (clf.predict_proba(sample), classnames_list)
        testimg_idx += 1
    actual = split.test_labels
    return (predicted_classes, predicted_values, actual)

ModelInfoByModelName['NuSVC'] = [nu_svc,'C']


# ### Gradient Boosting Classifier

# In[ ]:

def grad_clf(train_classed_data, test_classed_data, classnames_list, split):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(train_classed_data, split.train_classed_labels)
    predicted_classes = clf.predict(test_classed_data)
    predicted_values = np.zeros(test_classed_data.shape[0])
    testimg_idx = 0
    for sample in test_classed_data:
        predicted_values[testimg_idx] = marg_prob_to_pred_value (clf.predict_proba(sample), classnames_list)
        testimg_idx += 1
    actual = split.test_labels
    return (predicted_classes, predicted_values, actual)

ModelInfoByModelName['GradClf'] = [grad_clf,'C']


# ### Decision Tree Classifier

# In[ ]:

def dec_tree_clf(train_classed_data, test_classed_data, classnames_list, split):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(train_classed_data, split.train_classed_labels)
    predicted_classes = clf.predict(test_classed_data)
    predicted_values = np.zeros(test_classed_data.shape[0])
    testimg_idx = 0
    for sample in test_classed_data:
        predicted_values[testimg_idx] = marg_prob_to_pred_value (clf.predict_proba(sample), classnames_list)
        testimg_idx += 1
    actual = split.test_labels
    return (predicted_classes, predicted_values, actual)

ModelInfoByModelName['DecTreeClf'] = [dec_tree_clf,'C']


# ### Random Forest Classifier

# In[ ]:

def rand_forest_clf(train_classed_data, test_classed_data, classnames_list, split):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators = 30)
    clf.fit(train_classed_data, split.train_classed_labels)
    predicted_classes = clf.predict(test_classed_data)
    predicted_values = np.zeros(test_classed_data.shape[0])
    testimg_idx = 0
    for sample in test_classed_data:
        predicted_values[testimg_idx] = marg_prob_to_pred_value (clf.predict_proba(sample), classnames_list)
        testimg_idx += 1
    actual = split.test_labels
    return (predicted_classes, predicted_values, actual)

ModelInfoByModelName['RandForClf'] = [rand_forest_clf,'C']


# ### K Neighbors Classifier

# In[ ]:

def k_neigh_clf(train_classed_data, test_classed_data, classnames_list, split):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=10, weights='distance')
    clf.fit(train_classed_data, split.train_classed_labels)
    predicted_classes = clf.predict(test_classed_data)
    predicted_values = np.zeros(test_classed_data.shape[0])
    testimg_idx = 0
    for sample in test_classed_data:
        predicted_values[testimg_idx] = marg_prob_to_pred_value (clf.predict_proba(sample), classnames_list)
        testimg_idx += 1
    actual = split.test_labels
    return (predicted_classes, predicted_values, actual)

ModelInfoByModelName['KNeighClf'] = [k_neigh_clf,'C']


# ### AdaBoost Classifier

# In[ ]:

def ada_clf(train_classed_data, test_classed_data, classnames_list, split):
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=50)
    clf.fit(train_classed_data, split.train_classed_labels)
    predicted_classes = clf.predict(test_classed_data)
    predicted_values = np.zeros(test_classed_data.shape[0])
    testimg_idx = 0
    for sample in test_classed_data:
        predicted_values[testimg_idx] = marg_prob_to_pred_value (clf.predict_proba(sample), classnames_list)
        testimg_idx += 1
    actual = split.test_labels
    return (predicted_classes, predicted_values, actual)

ModelInfoByModelName['AdaBstClf'] = [ada_clf,'C']


# ## Regression

# ### Lasso Regression

# In[ ]:

def lasso_regression(train_data, train_labels, test_data, test_labels):
    from sklearn import linear_model 
    from sklearn.linear_model import Lasso
    regr = linear_model.Lasso
    clf=regr()
    clf.fit(train_data, train_labels)
    predicted = clf.predict(test_data)
    actual = test_labels
    return (predicted, actual)

ModelInfoByModelName['LassoReg'] = [lasso_regression,'R']


# ### Linear Regression

# In[ ]:

def lin_reg(train_data, train_labels, test_data, test_labels):
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(train_data, train_labels)
    predicted = lin_reg.predict(test_data)
    actual = test_labels
    return (predicted, actual)

ModelInfoByModelName['LinReg'] = [lin_reg,'R']


# ### SVR

# In[ ]:

def svr(train_data, train_labels, test_data, test_labels):
    from sklearn.svm import SVR
    svr = SVR('linear')
    svr.fit(train_data, train_labels)
    predicted = svr.predict(test_data)
    actual = test_labels
    return (predicted, actual)

ModelInfoByModelName['SVR'] = [svr,'R']


# ### NuSVR

# In[ ]:

def nusvr(train_data, train_labels, test_data, test_labels):
    from sklearn.svm import NuSVR
    nusvr = NuSVR('rbf')
    nusvr.fit(train_data, train_labels)
    predicted = svr.predict(test_data)
    actual = test_labels
    return (predicted, actual)

ModelInfoByModelName['NuSVR'] = [nusvr,'R']


# ### Ridge Regression

# In[ ]:

def ridge(train_data, train_labels, test_data, test_labels, solver):
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=0.1, solver=solver)
    ridge.fit(train_data, train_labels)
    predicted = ridge.predict(test_data)
    actual = test_labels
    return (predicted, actual)

ModelInfoByModelName['RidgeReg'] = [ridge,'R']


# ### Elastic Net

# In[ ]:

def e_net(train_data, train_labels, test_data, test_labels):
    from sklearn.linear_model import ElasticNet
    e_net = ElasticNet(alpha=0.001, l1_ratio=0.5,)
    e_net.fit(train_data, train_labels)
    predicted = e_net.predict(test_data)
    actual = test_labels
    return (predicted, actual)

ModelInfoByModelName['ElasNetReg'] = [e_net,'R']


# ### Decision Tree Regressor

# In[ ]:

def dec_tree(train_data, train_labels, test_data, test_labels):
    from sklearn.tree import DecisionTreeRegressor
    tree = DecisionTreeRegressor()
    tree.fit(train_data, train_labels)
    predicted = tree.predict(test_data)
    actual = test_labels
    return (predicted, actual)

ModelInfoByModelName['DecTreeReg'] = [dec_tree,'R']


# ### Random Forest Regressor

# In[ ]:

def rand_forest(train_data, train_labels, test_data, test_labels):
    from sklearn.ensemble import RandomForestRegressor
    forest = RandomForestRegressor(n_estimators=30)
    forest.fit(train_data, train_labels)
    predicted = forest.predict(test_data)
    actual = test_labels
    return (predicted, actual)

ModelInfoByModelName['RandForReg'] = [rand_forest,'R']


# ### KNeighbors Regressor

# In[ ]:

def k_neigh(train_data, train_labels, test_data, test_labels):
    from sklearn.neighbors import KNeighborsRegressor
    neigh = KNeighborsRegressor(n_neighbors = 20, weights='distance', p=1)
    neigh.fit(train_data, train_labels)
    predicted = neigh.predict(test_data)
    actual = test_labels
    return (predicted, actual)

ModelInfoByModelName['KNeighReg'] = [k_neigh,'R']


# ### Gradient Boosting Regressor

# In[ ]:

def grad_boost(train_data, train_labels, test_data, test_labels):
    from sklearn.ensemble import GradientBoostingRegressor
    boost = GradientBoostingRegressor(n_estimators = 120)
    boost.fit(train_data, train_labels)
    predicted = boost.predict(test_data)
    actual = test_labels
    return (predicted, actual)

ModelInfoByModelName['GraBstReg'] = [grad_boost,'R']


# ### SGD Regressor

# In[ ]:

def sgd_reg(train_data, train_labels, test_data, test_labels):
    from sklearn.linear_model import SGDRegressor
    sgd = SGDRegressor(penalty='elasticnet', n_iter=20)
    sgd.fit(train_data, train_labels)
    predicted = sgd.predict(test_data)
    actual = test_labels
    return (predicted, actual)

ModelInfoByModelName['SGDReg'] = [sgd_reg,'R']


# ### AdaBoost (KNeigh) Regressor

# In[ ]:

def ada_boost_reg(train_data, train_labels, test_data, test_labels):
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.neighbors import KNeighborsRegressor
    ada_boost = AdaBoostRegressor(KNeighborsRegressor(), n_estimators = 10, loss='linear')
    ada_boost.fit(train_data, train_labels)
    predicted = ada_boost.predict(test_data)
    actual = test_labels
    return (predicted, actual)

ModelInfoByModelName['AdaBstReg'] = [ada_boost_reg,'R']


# ### Maps to convert between model names and model functions

# In[ ]:

ModelByModelName = {k:v[0] for k, v in ModelInfoByModelName.items()}
ModelNameByModel = {v[0]:k for k, v in ModelInfoByModelName.items()}
ModelTypeByModel = {v[0]:v[1] for k, v in ModelInfoByModelName.items()}


# ## Model Scoring

# ### Accuracy

# #### Average Accuracy

# In[ ]:

# Haven't ever used
# but will calculate accuracy from predicted class names and actual classed labels
def Accuracy(predicted_classes, actual):
    acc = []
    right_wrong = predicted_classes - actual
    for prediction in right_wrong:
        if prediction == 0:
            acc = np.append(acc, 1)
        else:
            acc = np.append(acc, 0)
    acc = np.mean(acc)
    return (acc)


# #### Per Class Accuracy

# In[ ]:

# Gives accuracy for each class
def Class_Acc(predictions, actual, num_classes, test_size):
    acc = []
    right_wrong = predictions - actual
    for prediction in right_wrong:
        if prediction == 0:
            acc = np.append(acc, 1)
        else:
            acc = np.append(acc, 0)
    class_acc = []
    index=0
    for i in range (0,num_classes):
        class_acc.append (acc[index : index+test_size])
        print (np.mean(acc[index : index+test_size]))
        index = index+test_size
    class_acc = np.mean(class_acc)
    print ('Average Accuracy')
    print (class_acc)


# ### Performance VS Feature Number

# In[ ]:

# Standard method of scoring regression performance using LDA and Pearson
def Score_Reg(model, num_features, split):
    scores = []
    lda_scores = []
    n_classes = split.get_n_classes()
    for num_feature in num_features:
        new_train = split.train_set[:,:num_feature]
        new_test = split.test_set[:,:num_feature]
        predictions, actual = model(new_train, split.train_labels, new_test, split.test_labels)# Standard method of scoring regression performance using LDA and Pearson

        score, p_value = pearsonr(predictions, actual)
        scores = np.append(scores, score)
        if num_feature>n_classes-2:
            try:
                lda_train, lda_test = lda(new_train, new_test, split)
                lda_predictions, lda_actual = model(lda_train, split.train_labels, lda_test, split.test_labels)
                lda_score, lda_p_value = pearsonr(lda_predictions, lda_actual)
                lda_scores = np.append(lda_scores,lda_score)
            except np.linalg.LinAlgError:
                pass
    scores_R = scores*scores
    lda_scores_R = lda_scores*lda_scores
    return(scores_R, lda_scores_R)


# In[ ]:

# Standard method of scoring regression performance using LDA and Pearson
def Score_Clf(model, num_features, class_vals, split):
    scores = []
    lda_scores = []
    n_classes = split.get_n_classes()
    for num_feature in num_features:
        new_train = split.train_set[:,:num_feature]
        new_test = split.test_set[:,:num_feature]
        pred_cls_names, predictions, actual = model(new_train, new_test, class_vals, split)
        score, p_value = pearsonr(predictions, actual)
        scores = np.append(scores, score)
        if num_feature>n_classes-2:
            try:
                lda_train, lda_test = lda(new_train, new_test, split)
                lda_pred_cls_names, lda_predictions, lda_actual = model(lda_train, lda_test, class_vals, split)
                lda_score, lda_p_value = pearsonr(lda_predictions, lda_actual)
                lda_scores = np.append(lda_scores,lda_score)
            except np.linalg.LinAlgError:
                pass
    scores_R = scores*scores
    lda_scores_R = lda_scores*lda_scores
    return(scores_R, lda_scores_R)


# In[ ]:

def Run_Model (model, train, test, split):
    if ModelTypeByModel[model] == 'C':
        cls_names, predictions, actual = model (train, test, split.get_class_vals(), split)
    else:
        predictions, actual = model (train, split.train_labels, test, split.test_labels)
    return (predictions, actual)


# In[ ]:

# Varies number of features, reports R^2 for model and model+LDA
# N.B.: Assumes features are already sorted by weight!
def Score_Model(model, feature_range, split):
    scores = np.array([None]*len(feature_range), dtype=np.float64)
    lda_scores = np.array([None]*len(feature_range), dtype=np.float64)
    idx = 0
    n_classes = split.get_n_classes()
    class_vals = split.get_class_vals()
    for feature_num in feature_range:
        new_train, new_test = split.get_trimmed_features (feature_num)
        predictions, actual = Run_Model (model, new_train, new_test, split)

        score, p_value = pearsonr(predictions, actual)
        scores[idx] = score*score
        if feature_num>n_classes-2:
            try:
                lda_train, lda_test = lda(new_train, new_test, split)
                lda_predictions, lda_actual = Run_Model (model, lda_train, lda_test, split)

                lda_score, lda_p_value = pearsonr(lda_predictions, lda_actual)
                lda_scores[idx] = lda_score*lda_score
            except np.linalg.LinAlgError:
                pass
        idx += 1

    return(scores, lda_scores)


# In[ ]:

def most_common(lst):
    return max(set(lst), key=lst.count)


# In[ ]:

def Average_Score (scores):
    return (np.nanmean(np.vstack(scores), axis=0))


# The following two functions are examples of how to save the results and scores of predictors.
# 
# Below these are several functions for graphing the figures referenced in the paper.

# In[ ]:

# Saves right into home directory
def Save_Results(model_name, fs_name, avg_score):
    scores_name_base = model_name
    out_name = scores_name_base+fs_name
    if not os.path.isdir(scores_name_base):
        os.mkdir (scores_name_base)
    with open(os.path.join(scores_name_base, out_name), 'wb') as name:
        np.savetxt(name, avg_score, delimiter=",")


# In[ ]:

# Saves into home/suned/Graphs/Scores/
def Save_Results_New(model_name, fs_name, avg_score):
    scores_name_base = 'Graphs/Scores/'+model_name
    out_name = model_name+fs_name
    if not os.path.isdir(scores_name_base):
        os.mkdir (scores_name_base)
    with open(os.path.join(scores_name_base, out_name), 'wb') as name:
        np.savetxt(name, avg_score, delimiter=",")


# In[ ]:

# Plots graph of num_features and scores
# Saves to home/suned/Graphs/Plots/
def Graph(title, num_features, avg_score, lda_avg_score):
    raw_score = (avg_score[len(avg_score)-1])
    num_feat_lda = num_features[len(num_features)-len(lda_avg_score):]
    pl.figure()
    plot_score, = pl.plot(num_features, avg_score, 'b', label="Without LDA")
    plot_lda_score, = pl.plot(num_feat_lda, lda_avg_score, 'g', label="With LDA (12 components)")
    plot_benchmark, = pl.plot(num_features, raw_score*np.ones(len(num_features)),'r', label="All Features")

    best_num_feat = num_features[np.nanargmax (avg_score)]
    best_num_feat_lda = num_feat_lda[np.nanargmax (lda_avg_score)]
    
   # max_point = pl.plot(best_num_feat, max(avg_score), 'bo', label="(Features, Max Score)")
  #  lda_max_point = pl.plot(best_num_feat_lda, max(lda_avg_score), 'go')
    max_coords = '('+str(best_num_feat)+', ' + str("%.4f" % (np.nanmax(avg_score)))+')'
    text_x = num_features[0]+(0.22*(num_features[-1]-num_features[0]))
    pl.text(text_x,0.52, 'No LDA (features, max score): '+max_coords)
    lda_max_coords = '('+str(best_num_feat_lda)+', '+str("%.4f" % (np.nanmax(lda_avg_score)))+')'
    pl.text(text_x,0.56, 'LDA (features, max score): '+lda_max_coords)
    pl.title(title)
    pl.xlabel('Number of Features')
    pl.ylabel('Coefficient of Determination '+'( $R^2$)')
    pl.ylim([0.5, 1.0])
    pl.xlim([num_features[0],num_features[-1]])
    #pl.xticks(num_features[1], num_features[-1]+1, 20)
    # pl.legend([plot_score, plot_lda_score, plot_benchmark], ["Without LDA", "With LDA (12 components)", "All Features"], bbox_to_anchor=(1.05, 1), loc=2)
    pl.savefig(os.path.join (Config.graphs_dir,title+'.pdf'), format='pdf', dpi=150)
    pl.show()
    


# In[ ]:

# Separate Graphing Mechanism for PCA
# Graphs and saves the scores
def Graph_PCA(title, num_features, avg_score, raw_score):
    pl.figure()
    plot_score, = pl.plot(num_features, avg_score, 'b')
    plot_benchmark, = pl.plot(num_features, raw_score*np.ones(len(num_features)), 'r')
    ordered_num_feat = [num_features for (avg_score,num_features) in sorted(zip(avg_score,num_features))]
    best_num_feat = ordered_num_feat[-1]
    max_coords = '('+str(best_num_feat)+', ' + str("%.4f" % (max(avg_score)))+')'
    pl.text(len(num_features)*0.22,0.52, '(features, max score): '+max_coords)
    pl.title(title)
    pl.xlabel('Number of Principal Components')
    pl.ylabel('Coefficient of Determination '+'( $R^2$)')
    pl.ylim([0.5, 1.0])
    pl.xlim([num_features[0],num_features[-1]])
 #   pl.legend([plot_score, plot_benchmark], ["PCA Performance", "All Features"], bbox_to_anchor=(1.05, 1), loc=2)
    pl.savefig(os.path.join (Config.graphs_dir,title+'.pdf'), format='pdf', dpi=1000)
    pl.show()


# In[ ]:

def Calculate_Aging_Rate(prediction, actual):
    rate = prediction/actual
    return (rate)


# In[ ]:

def Calculate_Relative_Age_Rate(prediction, actual):
    age_rate = (prediction - actual)/actual
    return (age_rate)


# In[ ]:

def Read_Scores(model_name, fs_name):
    try:
        scores_name_base = os.path.join (Config.graphs_dir,'Scores',model_name)
        out_name = model_name+fs_name
        lda_out_name = out_name+'_lda'
        scores = np.loadtxt(os.path.join(scores_name_base, out_name), delimiter = ",")
    except:
        scores_name_base = os.path.join (Config.graphs_dir,'Scores','SCORES',model_name)
        out_name = model_name+fs_name
        lda_out_name = out_name+'_lda'
        scores = np.loadtxt(os.path.join(scores_name_base, out_name), delimiter = ",")
    try:
        lda_scores = np.loadtxt(os.path.join(scores_name_base, lda_out_name), delimiter = ",")
        return (scores, lda_scores)
    except:
        return (scores)


# # General Function for Predicting Physiological Ages

# ##### Create_Aging_Scores is the main function that uses previous functions to create a dictionary of physiological ages organized by individuals in the SardiNIA dataset. The parameters given are number of training/testing splits to be done, model name (see functions), waves of SardiNIA dataset used.

# In[ ]:

# Regression strategy based on "Best Aging Rates" notebook
# Works for classifier and regressor models
# When read_splits is False, uses the split number as the random seed.
def Create_Aging_Scores (num_splits, model, waves = range (1,4), read_splits = True):
    aging_scores = {}
    pearsons = {}
    # Change to range(1,4) for all 3 waves
    steps = (num_splits*len(waves))+len(waves)
    step = 0
    p = ProgressBar(steps)
    for wavenum in waves:
        p.animate (step)
        step +=1
        wi = Config.wave_info (wavenum)
        wd = Config.wave_data (wavenum)
        Split.filename_base = wi.filename_base
        best_num_feat = wi.best_models[ModelNameByModel[model]]
        for split_num in range(num_splits):
            p.animate (step)
            step +=1
            split = Split()
            if (read_splits):
                split.read(split.get_fname (split_num))
                mrmr_weights = read_mrmr(wavenum, split_num)
            else:
                # The last optional parameter is a one-time random number seed for the permute function
                # The mrmrs were computed for splits where the random number seed was the split number
                split.load_wave_train_test_RS (wavenum, split_num)
                mrmr_weights = read_mrmr(wavenum, split_num)

            split.norm_weigh_sort(mrmr_weights)
            new_train, new_test = split.get_trimmed_features (best_num_feat)

            try:
                lda_train, lda_test = lda(new_train, new_test, split)
                predictions, actual = Run_Model (model, lda_train, lda_test, split)

                n_test_samples = len(split.test_classed_labels)
                if not wavenum in pearsons:
                    pearsons[wavenum] = []
                pearsons[wavenum].append(pearsonr(predictions, actual))
                for test_idx in range(n_test_samples):
                    sample_id = split.test_id[test_idx]
                    if not sample_id in aging_scores:
                        aging_scores[sample_id] = {}
                    wave = 'wave_'+str(wavenum)
                    if not wave in aging_scores[sample_id]:
                        aging_scores[sample_id][wave] = {'A':actual[test_idx], 'P':[]}
                    aging_scores[sample_id][wave]['P'].append(predictions[test_idx])
            except np.linalg.LinAlgError as err:
                print ('split',split.get_fname (wavenum),err)
                sys.stdout.flush()

    p.animate (step)
    for wavenum in pearsons.keys():
        print('R^2 Average, wave',wavenum,':', np.sum(np.square(pearsons[wavenum]))/len(pearsons[wavenum]))
        sys.stdout.flush()
    return (aging_scores)


# In[ ]:

def Filter_Features (num_splits):
    feat_dict = {}
    # (1,4) for all 3 waves
    for wavenum in range(1,4):
        if wavenum == 1:
            Split.filename_base = "sard_w1_split_120tr_13te"
            raw_data = read_data_updated ('newWave1.csv', 'features')
            ages = raw_data.Age.values
            data_matrix, col_names, age_col_idx, id_col_idx = clean_convert(raw_data)
            col_names = np.delete(col_names, [age_col_idx, id_col_idx])
        if wavenum == 2:
            Split.filename_base = "sard_w2_split_96tr_10te"
            raw_data = read_w2 ('newWave2.csv')
            ages = raw_data.Age.values
            data_matrix, col_names, age_col_idx, id_col_idx = clean_convert(raw_data)
            col_names = np.delete(col_names, [age_col_idx, id_col_idx])
            print (len(col_names))
        if wavenum == 3:
            Split.filename_base = "sard_w3_split_82tr_8te"
            raw_data = read_data_updated ('newWave3.csv', 'samples')
            ages = raw_data.Age.values
            data_matrix, col_names, age_col_idx, id_col_idx = clean_convert(raw_data)
            col_names = np.delete(col_names, [age_col_idx, id_col_idx])
        split = Split()
        fisher_weights = []
        pearson_weights = []
        mrmr_weights = []
        for i in range(num_splits):
            if i == 1:
                print(split.train_set.shape[1])
            split.read(split.get_fname (i))
            pearson_weights.append(Pearson(split))
            fisher_weights.append(Fisher(split))
            mrmr_weights.append(read_mrmr(wavenum, i))
        fisher_weights = np.mean(np.vstack(fisher_weights), axis=0)
        pearson_weights = np.mean(np.vstack(pearson_weights), axis=0)
        mrmr_weights = np.mean(np.vstack(mrmr_weights), axis=0)
        for feat_idx in range(len(col_names)):
            feat_name = col_names[feat_idx]
            if not feat_name in feat_dict:
                feat_dict[feat_name] = {}
            wave_name  = 'wave'+str(wavenum)
            if not wave_name in feat_dict[feat_name]:
                feat_dict[feat_name][wave_name] = {'P':[], 'F':[], 'M':[]}
                feat_dict[feat_name][wave_name]['P'].append(pearson_weights[feat_idx])
                feat_dict[feat_name][wave_name]['F'].append(fisher_weights[feat_idx])
                feat_dict[feat_name][wave_name]['M'].append(mrmr_weights[feat_idx])
    return (feat_dict)


# In[ ]:

def Normalize_Feature_Scores (feature_weights):
    norm_weights = np.divide(feature_weights,(np.ones(len(feature_weights))*max(feature_weights)))
    return (norm_weights)


# In[ ]:

def Find_Common_Features ():
    common_col_names_dict = {}
    common_col_names_list = []
    
    w1_data = read_data_updated ('newWave1.csv', 'features')
    w1_ages = w1_data.Age.values
    data_matrix, w1_col_names, w1_age_col_idx, w1_id_col_idx = clean_convert(w1_data)
    w2_data = read_w2 ('newWave2.csv')
    w2_ages = w2_data.Age.values
    data_matrix, w2_col_names, w2_age_col_idx, w2_id_col_idx = clean_convert(w2_data)
    w3_data = read_data_updated ('newWave3.csv', 'samples')
    w3_ages = w3_data.Age.values
    data_matrix, w3_col_names, w3_age_col_idx, w3_id_col_idx = clean_convert(w3_data)
    
    for feat_name in w1_col_names:
        if not feat_name in common_col_names_dict:
            common_col_names_dict[feat_name] = []
        common_col_names_dict[feat_name].append('1')
    
    for feat_name in w2_col_names:
        if not feat_name in common_col_names_dict:
            common_col_names_dict[feat_name] = []
        common_col_names_dict[feat_name].append('2')
        
    for feat_name in w3_col_names:
        if not feat_name in common_col_names_dict:
            common_col_names_dict[feat_name] = []
        common_col_names_dict[feat_name].append('3')
        
    for feat_name in common_col_names_dict:
        if len(common_col_names_dict[feat_name]) == 3:
            common_col_names_list.append(feat_name)
        
    return (common_col_names_list)


# In[ ]:

def Harmonize_Features (common_col_names, wavenum, split):
    if wavenum == 1:
        data = read_data_updated ('newWave1.csv', 'features')
        ages = data.Age.values
        data_matrix, col_names, age_col_idx, id_col_idx = clean_convert(data)
        new_col_names = np.delete(col_names, [age_col_idx, id_col_idx])
    elif wavenum == 2:
        data = read_w2 ('newWave2.csv')
        ages = data.Age.values
        data_matrix, col_names, age_col_idx, id_col_idx = clean_convert(data)
        new_col_names = np.delete(col_names, [age_col_idx, id_col_idx])
    elif wavenum == 3:
        data = read_data_updated ('newWave3.csv', 'samples')
        ages = data.Age.values
        data_matrix, col_names, age_col_idx, id_col_idx = clean_convert(data)
        new_col_names = np.delete(col_names, [age_col_idx, id_col_idx])
    else:
        return ('Wave Number is out of bounds: Use 1, 2, or 3')
    
    delete_idxs = []
    
    for idx, feat_name in enumerate(new_col_names):
        if not feat_name in common_col_names:
            delete_idxs.append(idx)
    
    harmonized_train_set = np.delete(split.train_set, delete_idxs, axis=1)
    harmonized_test_set = np.delete(split.test_set, delete_idxs, axis=1)
    
    return (harmonized_train_set, harmonized_test_set)


# ###ProgressBar
# ```python
# p = Progressbar(120)
# for i in range(1, 120+1):
#     p.animate(i)
# ```

# In[ ]:

import sys, time
try:
    from IPython.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False

class ProgressBar(object):
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 20
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        print ('\r', self, end="")
        sys.stdout.flush()
        self.update_iteration(iter + 1)
        if (iter + 1 > self.iterations):
            print ()

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete ' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] +             (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)

