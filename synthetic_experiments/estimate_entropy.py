#!/usr/bin/env python
# *-* encoding: utf-8 *-*

import torch
import numpy as np
from estimators.knife import MargKernel
from datasets.synthetic import DataGeneratorMulti, get_random_data_generator
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from torch.utils import data
import os
import argparse
from scipy.stats import multivariate_normal
from numpy.linalg import slogdet
import logging
import shutil
from os.path import join
from doe.util import PDF as Doe

import json
from types import SimpleNamespace
from tools import MultiSummaryWriter

logger = logging.getLogger(__name__)

figsize = (6, 6)

def _c(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x

def get_dataloader(data_generator, n_batchsize, n_iterations, cubic=False):
    training_dataset_np = data_generator.rvs(size=(n_batchsize * n_iterations,))
    if cubic:
        training_dataset_np = training_dataset_np**3
    if len(training_dataset_np.shape) == 1:
        training_dataset_np = training_dataset_np[:, None]
    training_dataset = torch.tensor(training_dataset_np)

    dataset = data.TensorDataset(training_dataset)
    dataloader = data.DataLoader(dataset,
                                 batch_size=n_batchsize,
                                 shuffle=True,
                                 num_workers=min(1, os.cpu_count() - 1))
    return dataloader

def main(args):

    settings = f"""
    Settings:
      CUDA:             {torch.cuda.is_available()!s}
      debug:            {args.debug!s}
      logdir:           {args.logdir!s}
      run_id:           {args.run_id!s}
      batchsize:        {args.batchsize!s}
      iterations:       {args.iterations!s}
      epochs:           {args.epochs!s}
      kernel_samples:   {args.kernel_samples!s}
      d:                {args.dimension!s}
      pdf:              {args.pdf!s}
      rho:              {args.rho:.2f}
      lr:               {args.lr:.2f}
      """
    if args.pdf != 'gaussian':
        settings += f"data_components:  {args.data_components!s}"

    logger.info(settings)

    n_batchsize = args.batchsize
    n_iterations = args.iterations
    n_epochs = args.epochs
    n_samples_kernel = args.kernel_samples

    log_dir = args.logdir
    if log_dir is None:
        log_dir = 'results'

    if args.run_id is None:
        run_id = datetime.utcnow().strftime("%Y-%m-%dT%H%M%S")
    else:
        run_id = args.run_id

    log_dir += f'/{run_id!s}'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    s_writer = MultiSummaryWriter(log_dir=log_dir)

    d = args.dimension
    if args.pdf == 'gaussian':
        #A = np.random.random((d, d))
        #cov = A @ A.transpose()
        cov = np.eye(d)
        #cov = np.kron(np.eye(d//2), np.array([[1.0, 0.5], [0.5,1.0]]))
        data_generator = multivariate_normal(cov=cov)
        data_generator.entropy = lambda: np.prod(slogdet(2*np.pi*np.exp(1)*cov))/2
    else:
        data_generator1 = get_random_data_generator(base_pdf=args.pdf,
                                                    number=args.data_components,
                                                   )
        data_generator = DataGeneratorMulti(data_generator1, d=d)


    dataloader = get_dataloader(data_generator,
                                n_batchsize=n_batchsize,
                                n_iterations=n_iterations,
                                cubic=args.cubic)

    samples = data_generator.rvs(size=(n_samples_kernel,))
    if d == 1 and len(samples.shape) < 2:
        samples = samples[:, None]
    args_adaptive = SimpleNamespace(optimize_mu=True,
                                    marg_modes=args.kernel_samples,
                                    batch_size=args.kernel_samples,
                                    use_tanh=args.use_tanh,
                                    init_std=1e-2,
                                    cov_diagonal='var',
                                    cov_off_diagonal='var',
                                    average='var',
                                    ff_residual_connection=False,
                                    ff_activation='tanh',
                                    ff_layer_norm=False,
                                    ff_layers=1,
                                    )
    G_adaptive = MargKernel(args_adaptive,
                            args.dimension,
                            None,
                            init_samples=_c(torch.tensor(samples)),
                            )
    _c(G_adaptive)

    args_schraudolph =  SimpleNamespace(optimize_mu=False,
                                        marg_modes=args.kernel_samples,
                                        batch_size=args.kernel_samples,
                                        use_tanh=args.use_tanh,
                                        init_std=1e-2,
                                        cov_diagonal='var',
                                        cov_off_diagonal='zero',
                                        average='fixed')
    G_schraudolph = MargKernel(args_schraudolph,
                               args.dimension,
                               None,
                               init_samples=_c(torch.tensor(samples)),
                               )
    _c(G_schraudolph)

    G_doe = Doe(args.dimension,
                'gauss')

    _c(G_doe)


    model_names = ('KNIFE', 'Schraudolph', 'DoE')
    models = (G_adaptive, G_schraudolph, G_doe)
    opts = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in models]

    true_entropy = data_generator.entropy()
    if args.cubic:
        if args.pdf == 'gaussian':
            true_entropy += (np.log(3) - np.euler_gamma - np.log(2)) * args.dimension
        else:
            assert False, 'Cubic is only possible with gaussian data'
    if args.pdf == 'gaussian':
        if cov.shape[0] == 1:
            x0 = np.linspace(-5*cov, 5*cov, 1000).squeeze()
            if args.cubic:
                y0 = 1/3 * np.abs(x0)**(-2/3) * data_generator.pdf(np.abs(x0)**(1/3))
            else:
                y0 = data_generator.pdf(x0)
        else:
            x0 = None
            y0 = None
    else:
        x0, y0 = data_generator.plot()

    if args.rho_steps <= 1:
        iter_delta = (n_epochs*n_iterations)+1
    else:
        iter_delta = ((n_epochs*n_iterations)//(args.rho_steps)) + 1

    try:
        for i_epoch in range(n_epochs):
            logger.info(f'Epoch {i_epoch+1!s}/{n_epochs!s} starting ...')
            data_iter = iter(dataloader)
            for i in tqdm(range(len(dataloader))):
                i += i_epoch * n_iterations
                if (i+1) % iter_delta == 0 and args.pdf == 'gaussian':
                    cov = cov * args.rho
                    data_generator = multivariate_normal(cov=cov)
                    dataloader = get_dataloader(data_generator,
                                                n_batchsize=n_batchsize,
                                                n_iterations=n_iterations,
                                                cubic=args.cubic)
                    data_iter = iter(dataloader)
                    if args.cubic:
                        if args.pdf == 'gaussian':
                            true_entropy += 3*args.dimension*np.log(args.rho)/2
                        else:
                            assert False, 'Cubic only supported for gaussian pdf'
                    else:
                        true_entropy += args.dimension * np.log(args.rho) / 2
                x = next(data_iter)
                for o in opts:
                    o.zero_grad()
                x = x[0]
                x = _c(x)
                for model, opt, name in zip(models, opts, model_names):
                    loss_adaptive = model(x)
                    loss_adaptive.backward()
                    loss_np = loss_adaptive.detach().cpu().numpy()
                    error_np = loss_np - true_entropy
                    s_writer.add_scalar(name, 'loss', loss_np, i)
                    s_writer.add_scalar(name, 'error', error_np, i)
                    opt.step()

                # oracle
                s_writer.add_scalar('true_entropy', 'loss', true_entropy, i)
                x_np = x.detach().cpu().numpy()
                if args.cubic:
                    y_oracle = args.dimension*np.log(3) + 2/3 * np.sum(np.log(np.abs(x_np)), axis=-1) - data_generator.logpdf(np.abs(x_np)**(1/3))
                else:
                    y_oracle = -data_generator.logpdf(x_np)
                h_oracle = np.nanmean(y_oracle)
                s_writer.add_scalar('oracle', 'loss', h_oracle, i)
                s_writer.add_scalar('oracle', 'error', h_oracle - true_entropy, i)

    finally:
        s_writer.to_csv()
        s_writer.to_json()

    logger.info("Finished training.")

    # Final estimate
    dataloader = get_dataloader(data_generator,
                                n_batchsize=n_batchsize,
                                n_iterations=n_iterations,
                                cubic=args.cubic)

    # kde_samples = samples_np.transpose().squeeze()
    # kde = gaussian_kde(training_dataset_np.transpose().squeeze())
    #
    # kde_estimate = kde.logpdf(kde_samples).mean()

    outputs = dict()
    for model, opt, name in zip(models, opts, model_names):
        output = []
        outputs[name] = output
        H_est = np.zeros((len(dataloader),))
        with torch.no_grad():
            logger.info(f'{name}: final evaluation on fresh dataset...')
            for i, x in enumerate(tqdm(dataloader)):
                x = _c(x[0])
                H_est[i] = model(x).mean().detach().cpu().numpy()
            H_est = H_est.mean()

        logger.info(f"  True Entropy:    {true_entropy:.4f}")
        logger.info(f"  Final Estimate:  {H_est:.4f}")
        logger.info(f"  ERRROR:          {H_est - true_entropy:.4f}")
        output.append(np.abs((H_est - true_entropy)))

        # with torch.no_grad():
        #     estim_logpdf = model.logpdf(_c(torch.zeros((1, args.dimension)))).mean().cpu().numpy()
        # real_logpdf = data_generator.logpdf(np.zeros((args.dimension,)))
        # sq_pdf_zero_err = (np.exp(real_logpdf) - np.exp(estim_logpdf))**2/np.exp(real_logpdf)**2
        # logger.info(f'ERROR at zero:  {sq_pdf_zero_err:.4f}')
        # output.append(sq_pdf_zero_err)

        # logger.info(f"KDE Estimate:   {kde_estimate:.4f}")
        # logger.info(f"KDE ERRROR:     {kde_estimate - true_entropy:.4f}")

    if args.plot:

        for model, opt, name in zip(models, opts, model_names):
            if not hasattr(model, 'weigh'):
                continue
            # Weight dist
            with torch.no_grad():
                w = torch.softmax(model.weigh, dim=-1).detach().cpu().numpy().squeeze()
                x = model.means.detach().cpu().numpy().squeeze()
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.stem(x, w)
            ax.set_title('weights')
            ax.grid()
            s_writer.add_figure(name, 'weights', fig, 0)

            # Variances
            with torch.no_grad():
                logvar = model.logvar
                if model.use_tanh:
                    logvar = logvar.tanh()
                var = logvar.exp()
                cov = var.detach().cpu().numpy().squeeze()
                cov = cov ** (-2)
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.stem(x, cov)
            ax.grid()
            ax.set_title('var')
            s_writer.add_figure(name, 'var', fig, 0)

        # Plot estimated pdf
        x_np = np.linspace(np.min(x0), np.max(x0), 1000).reshape((-1, 1))
        x = _c(torch.tensor(x_np))



        logger.debug("Plotting adaptive estimation...")
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        pdf_plot_data = {}
        pdf_plot_data['x'] = x_np.squeeze().tolist()
        for model, opt, name in zip(models, opts, model_names):
            logger.debug(f"{name}: Getting PDF ...")
            with torch.no_grad():
                y1 = model.logpdf(x).exp()
                y1 = y1.detach().cpu().numpy()
            ax.plot(x_np, y1, label=name)
            pdf_plot_data[f'y_{name}'] = y1.squeeze().tolist()
        if x0 is not None:
            # Plot actual PDF
            ax.plot(x0, y0, label='True PDF')
            pdf_plot_data['x_true'] = x0.squeeze().tolist()
            pdf_plot_data['y_true'] = y0.squeeze().tolist()
        s_writer.add_plot('KNIFE', 'pdfs', pdf_plot_data, global_step=0)

        # kde_y_np = kde.pdf(x_np.transpose().squeeze())
        # ax.plot(x_np, kde_y_np, label='KDE PDF')
        ax.grid()
        ax.legend()
        ax.set_title('PDFs')
        s_writer.add_figure('KNIFE', 'pdfs', fig, 0)
        plt.show()

    return outputs

def meta_main(args):
    log_dir = f"{args.logdir!s}/pdf-{args.pdf!s}/cubic-{args.cubic!s}/d-{args.dimension!s}/rho-{args.rho:.2f}"
    args.logdir = log_dir
    os.makedirs(log_dir, exist_ok=True)

    NOW = datetime.utcnow().strftime("%Y-%m-%dT%H%M%S")
    loglevel = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=loglevel)

    filelogger = logging.FileHandler(join(log_dir, f"{NOW}.log"))
    logging.getLogger().addHandler(filelogger)
    try:

        all_outputs = dict()
        for i in range(args.reps):
            logger.info(f"Doing run {i + 1!s}...")
            if args.reps > 1:
                args.run_id = i + 1
            outputs = {k: np.array([v]) for k, v in main(args).items()}
            for name, value in outputs.items():
                if name not in all_outputs:
                    all_outputs[name] = value
                else:
                    all_outputs[name] = np.append(all_outputs[name], value, axis=0)

        logger.info('FINAL:')
        summary = {}
        for name, outputs in all_outputs.items():
            logger.info(f"{name}:")
            summary[name] = []
            for i in range(outputs.shape[1]):
                summary[name].append((outputs.mean(axis=0)[i], outputs.std(axis=0)[i]))
                logger.info(f'  error {i + 1}: {outputs.mean(axis=0)[i]:.4f} ± {outputs.std(axis=0)[i]:.4f}')
        summary_json = join(log_dir, 'summary.json')
        with open(summary_json, 'w') as fp:
            json.dump(summary, fp, indent=2)
    finally:
        logging.getLogger().removeHandler(filelogger)

    return log_dir

def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Train diff entropy')
    parser.add_argument('--reps', type=int, default=1, help='Repeat how ofter')
    parser.add_argument('--debug', action='store_true', required=False,
                        help='Enable debug output')
    parser.add_argument('--logdir', type=str, required=False, default='runs',
                        help='Root logging dir')
    parser.add_argument('--run_id', type=str, default=None,
                        required=False, help='tensorboard run id')
    parser.add_argument('--batchsize', type=int, required=False, default=128,
                        help='Size of batches')
    parser.add_argument('--iterations', type=int, required=False, default=1000,
                        help='Number of iterations per epoch')
    parser.add_argument('--epochs', type=int, required=False, default=1,
                        help='Number of epochs')
    parser.add_argument('--kernel_samples', type=int, required=False, default=500,
                        help='Number samples used for kernel construction')
    parser.add_argument('--rho', type=float, default=1.0,
                        help='multiplicative constant for stepping')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--rho_steps', type=int, default=1,
                        help='how many steps')
    parser.add_argument('--cubic', action='store_true', default=False,
                        help='Cubed transform')
    parser.add_argument('--use_tanh', action='store_true', default=False,
                        help='Use Tanh for the variance')

    parser.add_argument('--plot', required=False, action='store_true',
                        help='Plot the PDF')
    parser.add_argument('--pdf', type=str, choices=('triangle', 'affine', 'gaussian',),
                        required=False, default='triangle',
                        help='pdf type for matching')
    parser.add_argument('--dimension', type=int, required=False, default=5,
                        help='Data dimension')
    parser.add_argument('--data_components', type=int, required=False, default=15,
                        help='Number modes for the real pdf')

    args = parser.parse_args(argv)

    return args

if __name__ == '__main__':
    args = get_args()

    logging.getLogger('matplotlib').setLevel(logging.WARN)
    logging.getLogger().addHandler(logging.StreamHandler())
    meta_main(args)

