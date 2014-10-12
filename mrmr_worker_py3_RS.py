#!/usr/bin/env python3.4
#
# the split number is used as the seed for np.random.RandomState(split_num).permutation
# This way the splits themselves don't have to be stored.
import os, errno
import sys
import time
import pickle
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from Functions import Config, Split, mrmr, clean_convert, bin_data

# params: wave n_splits
if len(sys.argv) != 3:
	print ("Specify wave and number of splits:")
	print (os.path.basename(__file__),'1','5000')
	sys.exit(0)

wavenum = int (sys.argv[1])
nsplits = int (sys.argv[2])

wi = Config.wave_info(wavenum)
n_tr = wi.num_train
n_te = wi.num_test

wd = Config.wave_data(wavenum)
ages = wd.raw_data.Age.values

mrmr_filename_base = Config.wave_info(wavenum).filename_base
mrmr_filename_base += Config.mrmr_sfx
mrmr_dir = os.path.join(Config.data_dir(),mrmr_filename_base)

if not os.path.isdir(mrmr_dir):
	print ("crating directory '{}'".format(mrmr_dir))
	os.mkdir (mrmr_dir)

for splitnum in range(nsplits):
	# Read splits from files
	split = Split()

	mrmr_file_name = mrmr_filename_base+Config.split_num_fmt.format(splitnum)+".pickle"
	mrmr_file_path = os.path.join(mrmr_dir,mrmr_file_name)

	try:
		fd = os.open( mrmr_file_path, os.O_RDWR|os.O_CREAT|os.O_EXCL )
		split.train_test(n_tr, n_te, wd.data_class_views, wd.class_vals, wd.age_col_idx, wd.id_col_idx, splitnum)

		print ("opening {} for split {}".format(mrmr_file_path, splitnum))
		mrmr_values = mrmr(split)
		with open(mrmr_file_path, 'wb') as f:
			pickle.dump(mrmr_values, f)
		os.close(fd)
	except OSError as e:
		if e.errno != errno.EEXIST:
			print("fatal error",'split',splitnum, file=sys.stderr)
			raise e
		pass
	except Exception as e:
		print("fatal error",'split',splitnum, file=sys.stderr)
		raise e

