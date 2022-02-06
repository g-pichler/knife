import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join, dirname
import json
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
from collections import defaultdict
from itertools import chain
import argparse

PLOTSIZE_X = 11.7
PLOTSIZE_Y = 8.55

class MultiSummaryWriter(dict):
    df: pd.DataFrame

    def __init__(self, *args, **kwargs):
        super(MultiSummaryWriter, self).__init__()
        if 'log_dir' in kwargs:
            self.log_dir = kwargs['log_dir']
            del kwargs['log_dir']
        else:
            self.log_dir = args[0]
            args = args[1:]
        self.args = args
        self.kwargs = kwargs
        self.writers = dict()
        self.df = pd.DataFrame()

    def __getitem__(self, item):
        if item in self:
            return super(MultiSummaryWriter, self).__getitem__(item)
        else:
            new = SummaryWriter(join(self.log_dir, item),  *self.args, **self.kwargs)
            self[item] = new
            return new

    def add_scalar(self, item, name, value, global_step=0):
        self[item].add_scalar(name, value, global_step=global_step)
        self.df.loc[global_step, f'{name}_{item}'] = float(value)

    def add_figure(self, item, name, value: plt.Figure, global_step=0):
        self[item].add_figure(name, value, global_step=global_step)
        value.savefig(join(self.log_dir, f"{name}_{item}_{global_step!s}.png"))

    def add_plot(self, item, name, value, global_step):
        with open(join(self.log_dir, f"{name}_{item}_{global_step!s}.json"), 'w') as fp:
            json.dump(value, fp, indent=2)

    def to_csv(self):
        self.df.to_csv(join(self.log_dir, 'run.csv'))

    def to_json(self):
        self.df.to_json(join(self.log_dir, 'run.json'), indent=2)


class MultiOptimizer:
    def __init__(self, *optimizers):
        self.opts = optimizers
    def zero_grad(self):
        for opt in self.opts:
            opt.zero_grad()
    def step(self):
        for opt in self.opts:
            opt.step()





def plot_entropy(filename, show=False, save=True):
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5, 'lines.markersize': 10})
    color_palette = sns.color_palette("colorblind", 4)

    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    data_to_plot = {}
    for key, value in data.items():
        data_to_plot[key] = [v for k, v in value.items()]
    data_to_plot['x'] = range(len(data_to_plot['loss_KNIFE']))

    fig, ax = plt.subplots(figsize=(PLOTSIZE_Y, PLOTSIZE_Y))  # forward = False
    fig.set_figheight(PLOTSIZE_Y)
    fig.set_figwidth(PLOTSIZE_X)
    plt.step(list(data_to_plot['x']), data_to_plot['loss_KNIFE'], '-.', label='KNIFE', color=color_palette[0],
             alpha=0.7, linewidth=6)
    plt.step(list(data_to_plot['x']), data_to_plot['loss_Schraudolph'], '-.', label='Schrau.', color=color_palette[1],
             alpha=0.7, linewidth=6)
    plt.step(list(data_to_plot['x']), data_to_plot['loss_DoE'], '-.', label='DoE', color=color_palette[2],
             alpha=0.7, linewidth=6)
    plt.step(list(data_to_plot['x']), data_to_plot['loss_true_entropy'], label='True', color="black",
             linewidth=6)
    # plt.plot(list(data_to_plot['x']), data_to_plot['loss_oracle'], '-.', label='True', color=color_palette[3],
    #         linewidth=3)

    plt.xlabel('Iterations', fontsize=60)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    xmin, xmax = np.min(data_to_plot['x']), np.max(data_to_plot['x'])
    ax.set_xticks(np.round(np.linspace(0, xmax + 1, 5), 2))
    plt.ylabel('Differential Entropy', fontsize=60)

    # plt.grid(axis='x', color='0.95')
    leg = plt.legend(prop={'size': 40})
    leg.get_lines()[0].set_linewidth(10)
    leg.get_lines()[1].set_linewidth(10)
    leg.get_lines()[2].set_linewidth(10)

    plt.tight_layout()
    if save:
        plt.savefig(filename.replace('.json', '.pdf'), format='pdf')
    if show:
        plt.show()


def plot_pdf(filename, pdf_out=None, show=False, save=True):
    if pdf_out is None:
        pdf_out = filename.replace('.json', '.pdf')

    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5, 'lines.markersize': 10})
    color_palette = sns.color_palette("colorblind", 4)

    with open(filename, 'r') as json_file:
        data = json.load(json_file)
    data_to_plot = {}
    for key, value in data.items():
        data_to_plot[key] = value

    fig, ax = plt.subplots(figsize=(PLOTSIZE_X, PLOTSIZE_Y))  # forward = False
    plt.locator_params(axis="x", nbins=3)
    fig.set_figheight(PLOTSIZE_Y)
    fig.set_figwidth(PLOTSIZE_X)
    idx = 0
    for k, v in data_to_plot.items():
        if k.startswith('y_'):
            name = k[2:]
            x = 'x'
            lbl = name
            if name == 'Schraudolph':
                lbl = 'Schrau.'
            elif name == 'true':
                continue
            plt.plot(list(data_to_plot[x]), v, '-', label=lbl, color=color_palette[idx], alpha=0.7,
                     linewidth=10)
            idx += 1
    plt.plot(list(data_to_plot['x_true']), data_to_plot['y_true'], label='True', color='black',
             linewidth = 1)

    xmin, xmax = ax.get_xlim()
    ax.set_xticks(np.round(np.linspace(0, xmax, 3), 2))

    plt.xlabel('x', fontsize=60)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.ylabel('PDF', fontsize=60)
    # plt.grid(axis='x', color='0.95')
    custom_ticks = np.linspace(xmin, xmax, 3, dtype=int)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_ticks)
    leg = plt.legend(prop={'size': 40})
    leg.get_lines()[0].set_linewidth(10)
    leg.get_lines()[1].set_linewidth(10)
    leg.get_lines()[2].set_linewidth(10)
    plt.tight_layout()
    if save:
        plt.savefig(pdf_out, format='pdf')
    if show:
        plt.show()


def _get_runs(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)

    data_pd = {}
    model_list = ["club", "doe", "cpc", "mine", 'nwj', "knife"]
    for k, v in data.items():
        if k.startswith('estimate_'):
            data_pd[k] = v

    display_names = {x: x.upper() for x in model_list}
    display_names['cpc'] = 'InfoNCE'

    df = pd.DataFrame.from_dict(data_pd)

    return df, model_list, display_names

def collect_mi(filename):
    with open(filename, 'r') as json_file:
        data = json.load(json_file)

    basedir = dirname(filename)
    out_json = join(basedir, 'collected.json')

    for name, run_id in chain(data['best_run_id'].items(), zip(('oracle', 'true_mi'), (1,)*2)):
        with open(join(basedir, str(run_id), 'run.json'), 'r') as fp:
            run_data = json.load(fp)
        for k, v in run_data.items():
            if k == f'estimate_{name}':
                data[k] = v

    with open(out_json, 'w') as fp:
        json.dump(data, fp, indent=2)



def plot_mi_bias_var_mse(filename, pdf_out=None, save=True, show=False):
    if pdf_out is None:
        pdf_out = filename.replace('.json', '_bias_var_mse.pdf')

    df, model_list, display_names = _get_runs(filename)

    MIs = list(df.estimate_true_mi.unique())
    biases = defaultdict(list)
    vars = defaultdict(list)
    mses = defaultdict(list)
    for mi in MIs:
        mi_est_values = df[df[f'estimate_true_mi'] == mi]

        for model_name in model_list:
            model_estim = mi_est_values[f'estimate_{model_name}']
            est_mean = np.mean(model_estim)
            bias = np.abs(mi - est_mean)
            var = np.var(model_estim)
            biases[model_name].append(bias)
            vars[model_name].append(var)
            mses[model_name].append(bias ** 2 + var)

    plt.style.use('default')  # ('seaborn-notebook')

    colors = list(plt.rcParams['axes.prop_cycle'])
    col_idx = [2, 4, 5, 1, 3, 0]

    ncols = 1
    nrows = 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3. * nrows))
    axs = np.ravel(axs)

    for i, model_name in enumerate(model_list):
        plt.sca(axs[0])
        plt.plot(MIs, biases[model_name], label=display_names[model_name], marker='d',
                 color=colors[col_idx[i]]["color"])

        plt.sca(axs[1])
        plt.plot(MIs, vars[model_name], label=display_names[model_name], marker='d', color=colors[col_idx[i]]["color"])

        plt.sca(axs[2])
        plt.plot(MIs, mses[model_name], label=display_names[model_name], marker='d', color=colors[col_idx[i]]["color"])

    ylabels = ['Bias', 'Variance', 'MSE']
    for i in range(3):
        plt.sca(axs[i])
        plt.ylabel(ylabels[i], fontsize=15)

        if i == 0:
            pass
        if i == 1:
            plt.yscale('log')
        if i == 2:
            plt.legend(loc='upper left', prop={'size': 12})
            plt.xlabel('MI Values', fontsize=15)

    plt.gcf().tight_layout()
    if save:
        plt.savefig(pdf_out, bbox_inches='tight')
    if show:
        plt.show()


def plot_mi_run(filename, pdf_out=None, save=True, show=False):
    if pdf_out is None:
        pdf_out = filename.replace('.json', '.pdf')

    df, model_list, display_names = _get_runs(filename)

    total_steps = len(df['estimate_knife'])
    colors = sns.color_palette()

    EMA_SPAN = 200

    ncols = len(model_list)
    nrows = 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.4 * nrows))
    axs = np.ravel(axs)

    yaxis_mi = np.array(df[f'estimate_true_mi'])

    for i, model_name in enumerate(model_list):
        plt.sca(axs[i])
        p1 = plt.plot(np.array(df[f'estimate_{model_name}']), alpha=0.4, color=colors[0])[0]  # color = 5 or 0
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=4)
        mis_smooth = pd.Series(df[f'estimate_{model_name}']).ewm(span=EMA_SPAN).mean()
        mis_smooth = np.array(mis_smooth)

        if i == 0:
            plt.plot(mis_smooth, c=p1.get_color(), label='$\hat I$')
            plt.plot(yaxis_mi, color='k', label='True')
            plt.xlabel('Steps', fontsize=25)
            plt.ylabel('MI', fontsize=25)

            plt.legend(loc='upper left', prop={'size': 15})
        else:
            plt.plot(mis_smooth, c=p1.get_color())
            plt.yticks([])
            plt.plot(yaxis_mi, color='k')
            plt.xlabel('Steps', fontsize=25)

        plt.ylim(0, 1.55 * np.max(yaxis_mi))
        plt.xlim(0, total_steps)
        plt.title(display_names[model_name], fontsize=35)
        import matplotlib.ticker as ticker
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.EngFormatter())
        plt.xticks(horizontalalignment="right")
        # plt.grid()
        # plt.subplots_adjust( )

    plt.gcf().tight_layout()
    if save:
        plt.savefig(pdf_out, bbox_inches=None)
    if show:
        plt.show()
