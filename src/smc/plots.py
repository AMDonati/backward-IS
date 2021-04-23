import numpy as np
import os
import matplotlib.pyplot as plt


def plot_mean_square_error(dict_results, num_runs, out_folder, num_particles, backward_samples, args):
    if args.backward_is:
        backward_is_mean = [dict_results[k]["backward_is_mean"] for k in dict_results.keys()]
    if args.pms:
        pms_mean = [dict_results[k]["pms_mean"] for k in dict_results.keys()]
    fig, ax = plt.subplots(figsize=(25, 10))
    if args.pms:
        xx = np.linspace(1, len(pms_mean), len(pms_mean))
    if args.backward_is:
        xx = np.linspace(1, len(backward_is_mean), len(backward_is_mean))
    if args.backward_is:
        ax.plot(xx, backward_is_mean, color='blue', marker='x', label='backward IS Smoother')
    if args.pms:
        ax.plot(xx, pms_mean, color='red', label='PMS Smoother')
    labels = ['X_{}'.format(k) for k in list(dict_results.keys())]
    plt.xticks(ticks=xx, labels=labels)
    ax.grid('on')
    ax.legend(loc='upper center', fontsize=16)
    ax.set_title('mean squared error', fontsize=20)
    out_file = "mse_{}runs_{}particles_{}J".format(num_runs, num_particles, backward_samples)
    fig.savefig(os.path.join(out_folder, out_file))
    plt.close()

def plot_variance_error(dict_errors, num_runs, out_folder, num_particles, backward_samples, args):
    if args.backward_is:
        backward_is_var_1 = [np.var(dict_errors[k]["backward_runs"][:,0]) for k in dict_errors.keys()]
        backward_is_var_2 = [np.var(dict_errors[k]["backward_runs"][:,1]) for k in dict_errors.keys()]
    if args.pms:
        pms_var_1 = [np.var(dict_errors[k]["pms_runs"][:, 0]) for k in dict_errors.keys()]
        pms_var_2 = [np.var(dict_errors[k]["pms_runs"][:, 1]) for k in dict_errors.keys()]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12))
    if args.pms:
        xx = np.linspace(1, len(pms_var_1), len(pms_var_1))
    elif args.backward_is:
        xx = np.linspace(1, len(backward_is_var_1), len(backward_is_var_1))
    labels = ['X_{}'.format(k) for k in list(dict_errors.keys())]
    plt.sca(ax1)
    plt.xticks(ticks=xx, labels=labels)
    if args.backward_is:
        ax1.plot(xx, backward_is_var_1, color='blue', marker='x', label='Backward IS Smoother')
    if args.pms:
        ax1.plot(xx, pms_var_1, color='red', marker='x', label='PMS Smoother')
    plt.sca(ax2)
    plt.xticks(ticks=xx, labels=labels)
    if args.backward_is:
        ax2.plot(xx, backward_is_var_2, color='blue', marker='x')
    if args.pms:
        ax2.plot(xx, pms_var_2, color='red', marker='x')
    ax1.grid('on')
    ax2.grid('on')
    ax1.legend('upper center', fontsize=16)
    ax1.set_title('variance of the estimation error', fontsize=20)
    out_file = "error_variances_{}runs_{}particles_{}J".format(num_runs, num_particles, backward_samples)
    fig.savefig(os.path.join(out_folder, out_file))
    plt.close()

def plot_variance_square_error(dict_results, num_runs, out_folder, num_particles, backward_samples, args):
    if args.backward_is:
        backward_is_mean = [dict_results[k]["backward_is_var"] for k in dict_results.keys()]
    if args.pms:
        pms_mean = [dict_results[k]["pms_var"] for k in dict_results.keys()]
    fig, ax = plt.subplots(figsize=(25, 10))
    len_ = len(pms_mean) if args.pms else len(backward_is_mean)
    xx = np.linspace(1, len_, len_)
    if args.backward_is:
        ax.plot(xx, backward_is_mean, color='blue', marker='x', label='backward IS smoother')
    if args.pms:
        ax.plot(xx, pms_mean, color='red', marker='x', label='PMS smoother')
    labels = ['X_{}'.format(k) for k in list(dict_results.keys())]
    plt.xticks(ticks=xx, labels=labels)
    ax.legend('upper center', fontsize=16)
    out_file = "square_error_var_{}runs_{}particles_{}J".format(num_runs, num_particles, backward_samples)
    ax.grid('on')
    ax.set_title('variance of the squared error', fontsize=20)
    fig.savefig(os.path.join(out_folder, out_file))
    plt.close()

def plot_online_estimation_mse(dict_results, out_folder, args):
    if args.backward_is:
        backward_is = dict_results[0]["backward_by_seq"]
    if args.pms:
        pms = dict_results[0]["pms_by_seq"]
    fig, ax = plt.subplots(figsize=(30, 10))
    len_ = len(pms) if args.pms else len(backward_is)
    xx = np.linspace(1, len_, len_)
    if args.backward_is:
        ax.plot(xx, backward_is, color='blue', marker='x', label='backward IS smoother')
    if args.pms:
        ax.plot(xx, pms, color='red', marker='x', label='PMS smoother')
    labels = ['Y0:{}'.format(k) for k in range(1,len_+1)]
    plt.xticks(ticks=xx, labels=labels)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    out_file = "online_estimation_mse.pdf"
    ax.grid('on')
    ax.set_title(r'$\|X_0 - \mathbb{E}[X_0|Y_{0:j}]\|^2$ per sequence of observations $(Y_{0:j})_{1 \leq j \leq n}$', fontsize=20)
    fig.savefig(os.path.join(out_folder, out_file), format='pdf')
    plt.close()

def plot_online_estimation_mse_aggregated(pms_by_seq, backward_by_seq, out_folder):
    fig, ax = plt.subplots(figsize=(30, 10))
    xx = np.linspace(1, len(pms_by_seq), len(pms_by_seq))
    ax.plot(xx, backward_by_seq, color='tab:cyan', marker='x', label='backward IS smoother', lw=2)
    ax.plot(xx, pms_by_seq, color='salmon', marker='x', label='PMS smoother', lw=2)
    labels = ['Y0:{}'.format(k) for k in range(1,len(pms_by_seq)+1)]
    plt.xticks(ticks=xx, labels=labels, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    out_file = "online_estimation_mse.pdf"
    ax.grid(True,ls='--',lw =.5,c='k',alpha=.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=16)
    ax.set_title(r'$\|X_0 - \mathbb{E}[X_0|Y_{0:j}]\|^2$ per sequence of observations $(Y_{0:j})_{1 \leq j \leq n}$', fontsize=20)
    fig.savefig(os.path.join(out_folder, out_file), format='pdf')
    plt.close()

def plot_estimation_Xk(backward_all_k, pms_all_k, out_folder):
    fig, ax = plt.subplots(figsize=(30, 10))
    xx = np.linspace(1, len(backward_all_k), len(backward_all_k))
    ax.plot(xx, backward_all_k, color='tab:cyan', marker='x', label='backward IS smoother', lw=2)
    ax.plot(xx, pms_all_k, color='salmon', marker='x', label='PMS smoother', lw=2)
    labels = ['X_{}'.format(k) for k in range(0, len(pms_all_k))]
    plt.xticks(ticks=xx, labels=labels, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    out_file = "mse_all_Xk.pdf"
    ax.grid(True, ls='--', lw=.5, c='k', alpha=.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=20)
    ax.set_title(r'$\|X_k - \mathbb{E}[X_k|Y_{0:n}]\|^2$ for k $\in \{0,...,n\}$',
                 fontsize=20)
    fig.savefig(os.path.join(out_folder, out_file), format='pdf')
    plt.close()