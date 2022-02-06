import argparse
from tools import plot_pdf, plot_mi_run, plot_entropy, plot_mi_bias_var_mse

actions = {'pdf': plot_pdf,
           'mi': plot_mi_run,
           'entropy': plot_entropy,
           'mi_bias_var_mse': plot_mi_bias_var_mse}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make nice PDF plot')
    parser.add_argument('filename', type=str, help='JSON file to load')
    parser.add_argument('--show', action='store_true', help='Show plot')
    parser.add_argument('--type', choices=tuple(actions.keys()), default='entropy',
                        help='Type of plot')
    args = parser.parse_args()
    actions[args.type](args.filename, show=args.show)
