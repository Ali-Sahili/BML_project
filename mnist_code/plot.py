import matplotlib.pyplot as plt
import pickle
import numpy as np


def plot_org_results(
           precs=[1e-2, 2e-2, 5e-2, 1e-1, 2e-1,5e-1, 1e0, 2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2],
           plot_every = 10, num_epochs = 10
                 ):

    for prec in precs:

        file_path = 'results/original_results/vadam_mlp_class/mnist/act_func:relu|hidden_sizes:[400, 400]|prior_prec:' + str(prec) +'/batch_size:100|eval_mc_samples:10|num_epochs:200|seed:123|train_mc_samples:1/betas:(0.9, 0.999)|learning_rate:0.001|prec_init:'+str(prec)+'/metric_history.pkl'

        pkl_file = open(file_path, 'rb')
        metrics = pickle.load(pkl_file)
        pkl_file.close()

        num_evals = len(metrics['test_pred_logloss'])
        idx = np.arange(start = 0, stop = num_evals, step = plot_every)
        epoch = (idx+1) * num_epochs / num_evals
        met = np.array(metrics['test_pred_logloss'])

        if met.max() != 0:
           plt.plot( epoch, met[idx]/np.log(10), linestyle = '-', linewidth=2, 
                     label='vadam_'+str(prec))
           plt.legend()


def plot_results( bss = (1, 10, 100),
                  eps = (15, 15, 15),
                  prec = 5e1, plot_every = 10, num_epochs = 200):

    for bs, ep in zip(bss, eps):

        file_path = "results/our_results/bbb_mlp_class/mnist/act_func_relu_hidden_sizes_[400]_prior_prec_50.0/batch_size_"+ str(bs) +"_eval_mc_samples_10_num_epochs_15_seed_123_train_mc_samples_10/betas_(0.9, 0.999)_learning_rate_0.001_prec_init_50.0/metric_history.pkl"

        pkl_file = open(file_path, 'rb')
        metrics = pickle.load(pkl_file)
        pkl_file.close()

        num_evals = len(metrics['test_pred_logloss'])
        idx = np.arange(start = 0, stop = num_evals, step = plot_every)
        epoch = (idx+1) * num_epochs / num_evals
        met = np.array(metrics['test_pred_logloss'])
        print(met.shape, met.max())
  
        if met.max() != 0:
            plt.plot( epoch, met[idx]/np.log(10), linestyle = '-', linewidth=2, 
                      label=str(bs) +'ep_'+ str(ep))
            plt.legend()


    for bs, ep in zip(bss, eps):
        
        file_path = "results/our_results/vadam_mlp_class/mnist/act_func_relu_hidden_sizes_[400]_prior_prec_50.0/batch_size_"+ str(bs) +"_eval_mc_samples_10_num_epochs_15_seed_123_train_mc_samples_10/betas_(0.9, 0.999)_learning_rate_0.001_prec_init_50.0/metric_history.pkl"


        pkl_file = open(file_path, 'rb')
        metrics = pickle.load(pkl_file)
        pkl_file.close()

        num_evals = len(metrics['test_pred_logloss'])
        idx = np.arange(start = 0, stop = num_evals, step = plot_every)
        epoch = (idx+1) * num_epochs / num_evals
        met = np.array(metrics['test_pred_logloss'])

        if met.max() != 0:
            plt.plot( epoch, met[idx]/np.log(10), linestyle = '-', linewidth=2, 
                      label='vadam_'+ str(bs) +'ep_'+ str(ep))
            plt.legend()

plt.figure("original results")
plt.title("original results")
plot_org_results()

#plt.figure(figsize=(20, 20))
plt.figure("our experiments")
plt.title("our experiments")
plot_results()
plt.show()
