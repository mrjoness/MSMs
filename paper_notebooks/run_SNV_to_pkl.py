import sys
print(sys.executable)
import sklearn.preprocessing as pre, scipy, numpy as np, matplotlib.pyplot as plt, glob, pickle, pyemma as py, sys, os
import pandas as pd
import seaborn as sns 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

from temp_tf_load import *
sys.path.append('../')
from hde import HDE, analysis
import warnings
warnings.filterwarnings('ignore')


# load thermo and pairwise distance features seperately
# only pwd features are used to train the model, thermo are for plotting

skip_t = 1      # freq of skipping during training
skip_p = 1000  # freq of skipping during plotting (in dataframe)

n_epochs = 50
n_feat = 190

#npy_name = 'mdtraj-pwdr-AT-all-326T-395-1000-190.npy'
# reads in npy_name and prefix from sbatch 
npy_name = sys.argv[1]
prefix = sys.argv[2]

batch_size = int(sys.argv[3])
lag_time = int(sys.argv[4])
n_sm = int(sys.argv[5])        # number of leading slow modes to estimate
rep = int(sys.argv[6]) 

load_path = '/home/mikejones/scratch-midway2/srv/dna_data/' + npy_name

## set consistent value for number of 
pwd_features = np.load(load_path) # [:200]

pkl_path = './dataframe_outputs/' + prefix + '-' + npy_name.replace('.npy', '.pkl')

# scales all features to the range -1 and 1
scaler = pre.MinMaxScaler(feature_range=(-1, 1))
scaler.fit(np.concatenate(pwd_features))
pwd_features_s = [scaler.transform(item) for item in pwd_features] 

# reshape pwd_features
pwd_features_list = [traj for traj in pwd_features_s]

## comment out for skipping preferences, this gives 500 data points by default
skip_p = int(len(pwd_features) * len(pwd_features[0]) / 500)

from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min', restore_best_weights=True)

## establish hyperparameters for srv training
hde = HDE(n_feat, n_components=n_sm, lag_time=lag_time // skip_t, dropout_rate=0, batch_size=batch_size, n_epochs=n_epochs, 
          validation_split=0.2, batch_normalization=True, learning_rate = 0.01, reversible=True, 
          callbacks=[earlyStopping])

## format data for fitting
hde.r_degree = 2
hde.n_epochs = n_epochs  ## reset to 100
hde.fit(pwd_features_s)

## find hde coords and timescales
hde_coords = [hde.transform(item) for item in pwd_features]
hde_timescales = hde.timescales_

print(hde_timescales[:3])
print(hde.score(pwd_features_list, lag_time=lag_time))

pkl_list = np.append((hde_timescales[:3]), hde.score(pwd_features_list, lag_time=lag_time))

print(pkl_list)

pkl_path = 'hyper_test_out_30e_untrimmed/' + prefix + '-' + str(batch_size) + '-' + str(lag_time) + '-' + str(n_sm) + '-' + str(rep) + '.pkl'

with open(pkl_path, 'wb') as f:
    pickle.dump(pkl_list, f)


'''
## setup pandas data frame to compare srv coordinates to physical coordinates:
## edit 

skip = skip_p    #num coords to skip on plot
num_bp = 10


hde_coords_conc = np.concatenate(hde_coords)[::skip]
pwd_features_conc = np.concatenate(pwd_features)[::skip]

index_size = len(hde_coords_conc[:, 0])

print (np.shape(hde_coords_conc))
print (np.shape(pwd_features_conc))


hde_col_list = ['1st_EF', '2nd_EF', '3rd_EF', '4th_EF', '5th_EF', '6th_EF', '7th_EF', '8th_EF']
df = pd.DataFrame(data    = hde_coords_conc, 
                  columns = hde_col_list[:n_sm],
                  index   = np.linspace(1, index_size, index_size))


# try passing in all all pwd coords to pkl and make cvs post hoc
for i in range(len(pwd_features_conc[0, :])):
    df[str(i)] = pwd_features_conc[:, i]
    
## add hde_coords and filler zeros as last dataframe item   
zero_fill = np.zeros(len(pwd_features_conc) - n_sm)
df['hde_coords'] = np.append(hde_timescales, zero_fill)
    

df.to_pickle(pkl_path)

'''
