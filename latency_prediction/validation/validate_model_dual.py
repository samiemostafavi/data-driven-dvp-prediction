import warnings
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os
import math
import bisect
import tensorflow as tf
import warnings

# if you run python inside the folder, then:
sys.path.insert(0, '../lib')
print(sys.path)

from cde.data_collector import ParquetDataset
from cde.density_estimator import GPDExtremeValueMixtureDensityNetwork,NoNaNGPDExtremeValueMixtureDensityNetwork, NewGPDExtremeValueMixtureDensityNetwork
from cde.density_estimator import MixtureDensityNetwork
from cde.density_estimator import ExtremeValueMixtureDensityNetwork
from cde.density_estimator import plot_conditional_hist, measure_percentile, measure_percentile_allsame, measure_tail, measure_tail_allsame, init_tail_index_hill, estimate_tail_index_hill
from cde.evaluation.empirical_eval import evaluate_models_singlestate, empirical_measurer, evaluate_model_allstates, evaluate_models_allstates_plot, obtain_exp_value, evaluate_models_allstates_agg, evaluate_models_save_plots


# Path
path = '../training/saves/'
predictor_num = 3

""" load training data """
FILE_NAME = 'traindata_p'+str(predictor_num)+'_10k.npz'
npzfile = np.load(path + FILE_NAME)
train_data = npzfile['arr_0']
meta_info = npzfile['arr_1']
batch_size = int(meta_info[0])
ndim_x = int(meta_info[1])
predictor_num = int(meta_info[2])
p1_n_replicas = int(meta_info[3])
print('Predictor-%d training data loaded from .npz file. Rows: %d ' %(predictor_num,len(train_data[:,0,0])) , ' Columns: %d ' % len(train_data[0,:,0]), ' Replicas: %d' % len(train_data[0,0,:]) , ' ndim_x: %d' % ndim_x)
train_data = train_data[:,:,0]

""" import the test dataset into Numpy array """
file_addr = '../data/sim3hop_1_dataset_06_Sep_2021_11_20_40.parquet'
batch_size = 80000000
test_dataset = ParquetDataset(file_address=file_addr,predictor_num=predictor_num)
test_data = test_dataset.get_data_unshuffled(batch_size)
ndim_x_test = len(test_data[0])-1
#print(np.shape(train_data))
print('Predictor-%d dataset loaded from ' % predictor_num, file_addr,'. Rows: %d ' % len(test_data[:,0]), ' Columns: %d ' % len(test_data[0,:]), ' ndim_x: %d' % ndim_x_test)


""" load trained emm models """
FILE_NAME = 'trained_emm_p'+str(predictor_num)+'_s10_r0.pkl'
if not os.path.isfile(path + 'trained_models/' + FILE_NAME):
    print('No trained model found.')
    exit()
with open(path + 'trained_models/' + FILE_NAME, 'rb') as input:
    emm_model = NoNaNGPDExtremeValueMixtureDensityNetwork(name="a", ndim_x=ndim_x, ndim_y=1)
    emm_model._setup_inference_and_initialize()
    emm_model = pickle.load(input)
print(emm_model)


""" load trained gmm models """
FILE_NAME = 'trained_gmm_p'+str(predictor_num)+'_s10_r0.pkl'
if not os.path.isfile(path + 'trained_models/' + FILE_NAME):
    print('No trained model found.')
    exit()

with open(path + 'trained_models/' + FILE_NAME, 'rb') as input:
    gmm_model = MixtureDensityNetwork(name="b", ndim_x=ndim_x, ndim_y=1)
    gmm_model._setup_inference_and_initialize()
    gmm_model = pickle.load(input)

n_epoch = gmm_model.n_training_epochs

print(gmm_model)

""" Benchmark models on single states """
path = 'saves/'
warnings.filterwarnings('ignore')

plt.style.use('plot_style.txt')
if predictor_num is 1:
    cond_state = [1,1,2]
elif predictor_num is 2:
    cond_state = [1,1]
elif predictor_num is 3:
    cond_state = [1]

evaluate_models_save_plots(models=[emm_model,gmm_model],model_names=["EMM prediction","GMM prediction"],train_data=train_data,cond_state=cond_state,test_dataset=test_data,quantiles=[1-1e-1,1-1e-2,1-1e-3,1-1e-4,1-1e-5],save_fig_addr=path+'dual_test_')

#evaluate_models_singlestate(models=[model],model_names=["EMM"],train_data=train_data,cond_state=[1],test_dataset=test_data,quantiles=[1-1e-1,1-1e-2,1-1e-3,1-1e-5])
#evaluate_models_singlestate(models=[model],model_names=["EMM"],train_data=train_data,cond_state=[2],test_dataset=test_data,quantiles=[1-1e-1,1-1e-2,1-1e-3,1-1e-5])
#evaluate_models_singlestate(models=[model],model_names=["EMM"],train_data=train_data,cond_state=[10],test_dataset=test_data,quantiles=[1-1e-1,1-1e-2,1-1e-3,1-1e-5])

#unique_states,_,_ = get_most_common_unique_states(test_data[1000000:5000000,:],ndim_x=1,N=30,plot=True,save_fig_addr=path)
