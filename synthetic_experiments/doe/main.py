# Original Author: Karl Stratos (me@karlstratos.com)
# edited by Anonymous Author (2021)
import argparse
import copy
import inspect
import json
import math
import pathlib
import os

import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
from datetime import datetime
from os.path import exists
import shutil
from os.path import join


from collections import OrderedDict

from estimators.knife import KNIFE
from estimators.club import CLUB
from tools import MultiSummaryWriter, MultiOptimizer
from datasets import synthetic

import logging
logger = logging.getLogger(__name__)

DATETIME_NOW = datetime.utcnow().strftime("%Y-%m-%dT%H%M%S")


def control_weights(args, models):

    def init_weights(m):
        if hasattr(m, 'weight') and hasattr(m.weight, 'uniform_') and \
           args.init > 0.0:
            torch.nn.init.uniform_(m.weight, a=-args.init, b=args.init)

    for name in models:
        models[name].apply(init_weights)

    # Exactly match estimators for didactice purposes.
    if args.carry == 0:  # MINE(0) == DV
        models['mine'].fXY = copy.deepcopy(models['dv'].fXY)

    if args.alpha == 1:  # INTERPOL(1, *) == CPC
        models['interpol'].fXY = copy.deepcopy(models['cpc'].fXY)
        models['cpc'].transpose = True

    if args.alpha == 0 and args.a == 'e':  # INTERPOL(0, e) == NWJ
        models['interpol'].fXY = copy.deepcopy(models['nwj'].fXY)

def package(X, Y):
    return torch.cat([X.repeat_interleave(X.size(0), 0),
               Y.repeat(Y.size(0), 1)], dim=1)

def main(args):
    logger.info(args)
    if args.seed is None:
        args.seed = random.randint(1, 1000000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.cuda else 'cpu')
    ds = {'gaussian': synthetic.GaussianXZN,
          'uniform':  synthetic.UniformsXZN,
          'tri':      synthetic.MultiTriangleXZN}
    pXZN = ds[args.pdf](args.dim, device)

    s_writer = MultiSummaryWriter(log_dir=args.logdir)

    if args.I is not None:
        args.rho = float(pXZN.ItoRho(args.I))
    else:
        assert args.rho is not None, "either --rho or --I need to be given"
    rho = torch.tensor([args.rho], device=device, requires_grad=True)
    pXY = synthetic.XY(pXZN, rho, cubed=args.cubic)

    models = {
        'dv': util.SingleSampleEstimator(args, args.dim, args.hidden, args.layers,
                                         'dv'),
        'mine': util.MINE(args, args.dim, args.hidden, args.layers,
                          carry_rate=args.carry),
        'nwj': util.SingleSampleEstimator(args, args.dim, args.hidden, args.layers,
                                          'nwj'),
        'nwjjs': util.NWJJS(args, args.dim, args.hidden, args.layers),
        'cpc': util.CPC(args, args.dim, args.hidden, args.layers),
        'interpol': util.Interpolated(args, args.dim, args.hidden, args.layers,
                                      args.a, args.alpha),
        'doe': util.DoE(args, args.dim, args.hidden, args.layers, 'gauss'),
        'doe_l': util.DoE(args, args.dim, args.hidden, args.layers, 'logistic'),
        'club': CLUB(args, args.dim, args.dim),
        'knife': KNIFE(args, args.dim, args.dim),
    }
    for name in models:
        models[name] = models[name].to(device)
    control_weights(args, models)

    optims = {name: torch.optim.Adam(models[name].parameters(), lr=args.lr)
              for name in models if name != 'knife'}

    optims['knife'] = MultiOptimizer(torch.optim.Adam(models['knife'].kernel_marg.parameters(), lr=1e3*args.lr),
                                     torch.optim.Adam(models['knife'].kernel_cond.parameters(), lr=args.lr))

    train_MIs = {name: [] for name in models}

    I_delta = None
    if args.I_max is not None:
        I_delta = (args.I_max - args.I) / (args.rho_steps - 1)
    rho_iterations = args.steps // args.rho_steps + 1

    try:
        for step in range(1, args.steps + 1):
            if I_delta is not None and step % rho_iterations == 0:
                args.I += I_delta
                args.rho = pXZN.ItoRho(args.I)
                pXY.rho.data = torch.tensor([args.rho],
                                            requires_grad=True,
                                            dtype=torch.float,
                                            device=device)

            X, Y = pXY.draw_samples(args.N)
            XY_package = package(X, Y)
            L = {}
            for name in models:
                optims[name].zero_grad()
                if pXY.rho.grad is not None:
                    pXY.rho.grad.zero_()

                if hasattr(models[name], 'learning_loss'):
                    learning_loss = models[name].learning_loss(X.detach(), Y.detach())
                    mi = models[name](X.detach(), Y.detach())[0]
                    loss = learning_loss - (mi + learning_loss).detach()
                else:
                    loss = models[name](X.detach(), Y.detach(), XY_package.detach())

                L[name] = loss
                L[name].backward()
                train_MIs[name].append(-L[name].item())
                s_writer.add_scalar(name, 'estimate', -L[name].item(), global_step=step)
                s_writer.add_scalar(name, 'error', -L[name].item() - pXY.I(), global_step=step)
                s_writer.add_scalar(name, 'rel_error', (-L[name].item() - pXY.I()) / pXY.I(), global_step=step)

                nn.utils.clip_grad_norm_(models[name].parameters(), args.clip)
                optims[name].step()

                # Derivative of MI
                if hasattr(models[name], 'I'):
                    optims[name].zero_grad()
                    if pXY.rho.grad is not None:
                        pXY.rho.grad.zero_()
                    X, Y = pXY.repeat_samples()
                    MI = models[name].I(X, Y, package(X, Y))
                    MI.backward()
                    s_writer.add_scalar(name, 'dI', pXY.rho.grad.item(), global_step=step)

            # true derivative of MI
            s_writer.add_scalar('true_dI', 'dI', pXY.dI(), global_step=step)

            # Oracle (if implemented)
            if pXY.rho.grad is not None:
                pXY.rho.grad.zero_()
            try:
                logi = pXY.logi()
                Ioracle = logi.mean()
                Ioracle.backward()
                s_writer.add_scalar('oracle', 'dI', pXY.rho.grad.item(), global_step=step)
                s_writer.add_scalar('oracle', 'estimate', Ioracle, global_step=step)
                s_writer.add_scalar('oracle', 'error', Ioracle - pXY.I(), global_step=step)
                s_writer.add_scalar('oracle', 'rel_error', (Ioracle - pXY.I()) / pXY.I(), global_step=step)
            except NotImplementedError:
                pass

            # True MI and ln(N) for comparison
            s_writer.add_scalar('true_mi', 'estimate', pXY.I(), global_step=step)
            s_writer.add_scalar('ln N', 'estimate', np.log(args.N), global_step=step)

            log_line = 'step {:4d} | '.format(step)
            for name in L:
                log_line += '{:s}: {:6.2f} | '.format(name, -L[name])
            log_line += 'ln N: {:.2f} | I(X,Y): {:.2f}'.format(math.log(args.N), pXY.I().item())
            logger.info(log_line)
    finally:
        s_writer.to_csv()
        s_writer.to_json()
    # Final evaluation
    M = args.c * args.N
    X, Y = pXY.draw_samples(M)
    XY_package = torch.cat([X.repeat_interleave(M, 0), Y.repeat(M, 1)], dim=1)
    test_MI = {}
    for name in models:
        models[name].eval()
        if hasattr(models[name], 'I'):
            loss = -models[name].I(X, Y, XY_package)
        else:
            loss = models[name](X, Y, XY_package)
        test_MI[name] = -loss.item()

    logger.info('-'*150)
    log_line = 'Estimates on {:d} samples | '.format(M)
    for name in test_MI:
        log_line += '{:s}: {:6.2f} | '.format(name, test_MI[name])
    log_line += 'ln({:d}): {:.2f} | I(X,Y): {:.2f}'.format(M, math.log(M), pXY.I().item())
    logger.info(log_line)

    return test_MI, train_MIs, pXY.I().item()


def meta_main(args):

    if args.run_id is None:
        args.run_id = DATETIME_NOW
    log_dir = f'{args.logdir}/'
    log_dir += f'pdf-{args.pdf}/'
    log_dir += f'dim-{args.dim}/'
    log_dir += f'cubic-{args.cubic}/'
    log_dir += f'N-{args.N}/'

    os.makedirs(log_dir, exist_ok=True)
    loglevel = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(filename=join(log_dir, f"{DATETIME_NOW}.log"), level=loglevel)
    logging.getLogger('matplotlib').setLevel(logging.WARN)
    logging.getLogger().addHandler(logging.StreamHandler())

    hypers = OrderedDict({
        'hidden': [64, 128, 256],
        'layers': [1],
        'lr': [0.01, 0.003, 0.001, 0.0003],
        'init': [0.0, 0.1, 0.05],
        'clip': [1, 5, 10],
        'carry': [0.99, 0.9, 0.5],
        'alpha': [0.01, 0.5, 0.99],
        'a': ['e'],
        'seed': list(range(100000))
    })

    best_test_MI = {}
    best_train_MIs = {}
    bestargs = {}
    best_run_id = {}
    main_args = args
    for run_number in range(1, args.nruns + 1):
        args = copy.deepcopy(main_args)
        if args.nruns > 1:
            logger.info('RUN NUMBER: %d' % (run_number))
            for hyp, choices in hypers.items():
                choice = choices[torch.randint(len(choices), (1,)).item()]
                assert hasattr(args, hyp)
                args.__dict__[hyp] = choice
            args.logdir = log_dir + f'/{run_number}'
        else:
            args.logdir = log_dir + f'/{args.run_id}'

        if exists(args.logdir):
            shutil.rmtree(args.logdir)

        test_MI, train_MIs, mi = main(args)
        for name in test_MI:
            if run_number == 1:
                best_test_MI[name] = test_MI[name]
                best_train_MIs[name] = train_MIs[name]
                bestargs[name] = copy.deepcopy(args)
                best_run_id[name] = 1
            else:
                if math.isnan(best_test_MI[name]) or \
                   abs(mi - test_MI[name]) < abs(mi - best_test_MI[name]):
                    logger.info('*** New best test MI %g for %s from previous best %g'
                          % (test_MI[name], name, best_test_MI[name]))
                    best_test_MI[name] = test_MI[name]
                    best_train_MIs[name] = train_MIs[name]
                    bestargs[name] = copy.deepcopy(args)
                    best_run_id[name] = run_number

    M = args.c * args.N
    logger.info('-'*150)
    logger.info('Best test estimates on {:d} samples'.format(M))
    logger.info('-'*150)
    for name in best_test_MI:
        logger.info('{:10s}: {:6.2f} \t\t {:s}'.format(name, best_test_MI[name],
                                                 str(bestargs[name])))
    logger.info('-'*150)
    logger.info('ln({:d}): {:.2f}'.format(M, math.log(M)))
    logger.info('I(X,Y): {:.2f}'.format(mi))

    cukes = (args, mi, best_train_MIs, best_test_MI, bestargs)
    pickle.dump(cukes, open(args.pickle, 'wb'))

    with open(log_dir + '/summary.json', 'w') as fp:
        json.dump({'best_test_MI': best_test_MI,
                   'bestargs': {k: str(x) for k, x in bestargs.items()},
                   'best_run_id': best_run_id}, fp, indent=2)
    return(log_dir)


def get_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--N', type=int, default=64,
                        help='number of samples [%(default)d]')
    parser.add_argument('--c', type=int, default=1,
                        help='c*N is the number of samples for final '
                             'evaluation [%(default)g]')
    parser.add_argument('--rho', type=float, default=0.9,
                        help='correlation coefficient [%(default)g]')
    parser.add_argument('--dim', type=int, default=128,
                        help='number of dimensions [%(default)d]')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of hidden layers [%(default)d]')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate [%(default)g]')
    parser.add_argument('--init', type=float, default=0.0,
                        help='param init (default if 0) [%(default)g]')
    parser.add_argument('--clip', type=float, default=1,
                        help='gradient clipping [%(default)g]')
    parser.add_argument('--steps', type=int, default=1000, metavar='T',
                        help='number of training steps [%(default)d]')
    parser.add_argument('--carry', type=float, default=0.99,
                        help='EMA carry rate [%(default)g]')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='interpolation weight (on CPC term) [%(default)g]')
    parser.add_argument('--a', type=str, default='e', choices=['e', 'ff', 'lp'],
                        help='score function in TUBA, INTERPOL [%(default)s]')
    parser.add_argument('--nruns', type=int, default=1,
                        help='number of random runs (not random if set to 1) '
                             '[%(default)d]')
    parser.add_argument('--pickle', type=str, default='cukes.p',
                        help='output pickle file path [%(default)s]')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed [%(default)s]')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA?')

    # Custom Arguments
    parser.add_argument('--debug', action='store_true',
                        help='Debug loglevel')
    parser.add_argument('--logdir', type=str, default='./results',
                        help='Root logging directory')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Run ID for tensorboard logging; defaults to timestamp')
    parser.add_argument('--pdf', type=str, default='gaussian', choices=['gaussian', 'uniform', 'tri'],
                        help='PDF to use')
    parser.add_argument('--cubic', action='store_true',
                        help='Apply cubic transformation?')
    parser.add_argument('--I', type=float, default=None,
                        help='MI value to start')
    parser.add_argument('--I_max', type=float, default=None,
                        help='MI value to end')
    parser.add_argument('--rho_steps', type=int, default=5,
                        help='Number of steps for stepping up MI')

    # Arguments for KNIFE
    parser.add_argument('--tukey_transform', action='store_true', default=False,
                        help='tukey_transform')
    parser.add_argument('--no-optimize_mu', dest='optimize_mu', action='store_false',
                        default=True, help='Optimize means of kernel')
    parser.add_argument('--cond_modes', type=int,
                        default=128, help='Number of kernel components for conditional estimation')
    parser.add_argument('--use_tanh', action='store_true',
                        default=False, help='use tanh()')
    parser.add_argument('--init_std', type=float,
                        default=0.01, help='std for initialization')
    parser.add_argument('--cov_diagonal', type=str,
                        default='var', help='Diagonal elements of cov matrix different or the same?')
    parser.add_argument('--cov_off_diagonal', type=str,
                        default='var', help='Off-diagonal elements of cov matrix zero?')
    parser.add_argument('--average', type=str,
                        default='var', help='Weighted average?')
    parser.add_argument('--marg_modes', type=int,
                        default=128, help='Kernel components for marginal estimation')
    parser.add_argument('--ff_residual_connection', action='store_true',
                        default=False, help='FF Residual Connections')
    parser.add_argument('--ff_activation', type=str,
                        default='tanh', help='FF Activation Function')
    parser.add_argument('--ff_layer_norm', default=False,
                        action='store_true', help='Use a NormLayer?')

    args = parser.parse_args(argv)
    args.ff_layers = args.layers
    args.hidden = args.dim
    return args

if __name__ == '__main__':
    args = get_args()
    meta_main(args)
