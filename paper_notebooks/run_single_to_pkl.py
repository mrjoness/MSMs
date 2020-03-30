import sys, pickle #, cPickle
print(sys.executable)
import sklearn.preprocessing as pre, scipy, numpy as np, matplotlib.pyplot as plt, glob, pyemma as py, sys, os
import pandas as pd, seaborn as sns, argparse
from sklearn.model_selection import train_test_split

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
plot_num = 5000  # approximate number of points saved in dataframe and plottded

n_epochs = 100
n_feat = 190

#npy_name = 'mdtraj-pwdr-AT-all-326T-395-1000-190.npy'
# reads in npy_name and prefix from sbatch 

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)      #name of data densor saved in dna_outputs
parser.add_argument('--prefix', type=str)    #name of prefix based on specific run
parser.add_argument('--batch_size', type=int)
parser.add_argument('--lag_time', type=int)
parser.add_argument('--n_sm', type=int)
parser.add_argument('--n_epochs', type=int)
parser.add_argument('--n_feat', type=int)     #number of features per frame (190)
parser.add_argument('--n_plot', type=int)     #number of plot points to save to df 
parser.add_argument('--reversible', type=bool)    # if true, obeys detail balance
parser.add_argument('--trim_start', type=int)    # index where trim starts
parser.add_argument('--trim_end', type=int)    # index where trim ends
parser.add_argument('--n_traj', type=int)    # index where trim ends
args = parser.parse_args()

npy_name = args.name
prefix = args.prefix
batch_size = args.batch_size
lag_time = args.lag_time
n_sm = args.n_sm
n_epochs = args.n_epochs
n_feat = args.n_feat
plot_num = args.n_plot
reversible = args.reversible
trim_start = args.trim_start
trim_end = args.trim_end
n_traj = args.n_traj 

load_path = '/home/mikejones/scratch-midway2/srv/dna_data/' + npy_name
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

## set consistent value for number of 
#pwd_features = np.load(load_path)
pwd_features = np.load(load_path)[:n_traj, trim_start:trim_end]
print(np.shape(pwd_features))

pkl_path = './dataframe_outputs/' + prefix + '-' + npy_name.replace('.npy', '.pkl')
pkl_hde_path = './hde_outputs/' + prefix + '-' + npy_name.replace('.npy', '.pkl')

# scales all features to the range -1 and 1
scaler = pre.MinMaxScaler(feature_range=(0, 1))
#scaler.fit([item for item in pwd_features])
scaler.fit(np.concatenate(pwd_features))

# reshapes features into scaled list
pwd_features_s = [scaler.transform(item) for item in pwd_features] 

pwd_features_unscaled = [item for item in pwd_features] 

## comment out for skipping preferences, this gives 500 data points by default
skip_p = int(len(pwd_features) * len(pwd_features[0]) / plot_num)

from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min', restore_best_weights=True)

## try manually splitting data
train_data, test_data = train_test_split(pwd_features_s, test_size=0.5)

## establish hyperparameters for srv training
hde = HDE(n_feat, n_components=n_sm, lag_time=lag_time // skip_t, dropout_rate=0, batch_size=batch_size, n_epochs=n_epochs, 
          validation_split=0.2, batch_normalization=True, learning_rate = 0.01, reversible=reversible, 
          callbacks=[earlyStopping])

## format data for fitting
hde.r_degree = 2
hde.n_epochs = n_epochs  ## reset to 100
#hde.fit(train_data)
hde.fit(pwd_features_s)

## find hde coords and timescales
hde_coords = [hde.transform(item) for item in pwd_features]
hde_timescales = hde.timescales_

## calculate train and test scores for given # of slow modes
train_score = hde.score(train_data)
test_score = hde.score(test_data)

## setup pandas data frame to compare srv coordinates to physical coordinates:
## edit 

num_bp = 10

hde_coords_conc = np.concatenate(hde_coords)[::skip_p]
pwd_features_conc = np.concatenate(pwd_features)[::skip_p]

index_size = len(hde_coords_conc[:, 0])

print (np.shape(hde_coords_conc))
print (np.shape(pwd_features_conc))


hde_col_list = ['1st_EF', '2nd_EF', '3rd_EF', '4th_EF', '5th_EF', '6th_EF', '7th_EF', '8th_EF']
df = pd.DataFrame(data    = hde_coords_conc, 
                  columns = hde_col_list[:n_sm],
                  index   = np.linspace(1, index_size, index_size))


# try passing in all pwd coords to pkl and make cvs post hoc
for i in range(len(pwd_features_conc[0, :])):
    df[str(i)] = pwd_features_conc[:, i]
    
## add hde_coords and filler zeros as last dataframe item   
zero_fill = np.zeros(len(pwd_features_conc) - n_sm)
df['hde_coords'] = np.append(hde_timescales, zero_fill)

print(hde_timescales)
print(train_score, test_score)

zero_fill = np.zeros(len(pwd_features_conc) - 2)
df['train_test'] = np.append(np.array([train_score, test_score]), zero_fill)

df.to_pickle(pkl_path)

with open(pkl_hde_path, 'wb') as pickle_file:
    pickle.dump(hde, pickle_file)