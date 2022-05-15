import numpy as np
import os
from datetime import datetime

# imports for plotting
import matplotlib.pyplot as plt
from matplotlib import dates as md


images_path = '../images/'
def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    """
    Saves the images of the generated plots as png with a resolution of 300dpi.

    :param fig_id: file name
    :param tight_layout: True or False
    :param fig_extension: file format
    :param resolution: resolution of the plot
    :return: saved version of the plot
    """

    path = os.path.join(images_path, fig_id + '.' + fig_extension)
    print('Saving figure: ', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def model_summary(net):
    """
    :param net: neural net to print the summary
    :return: summary of the model parameters
    """

    SPACING = 40
    print('Model summary')
    print('-'*(SPACING*3))
    print(f"{'Layer Name':40s}\t{'Layer Shape':40s}\t{'# Parameters':40s}")
    print('-'*(SPACING*3))
    sum_params = 0
    params = []
    for name, param in net.named_parameters():
      if param.requires_grad:
            dimension = str(tuple(param.data.shape))
            print(f'{name:40s}\t{dimension:40s}\t{param.data.numel():<40d}')
            sum_params += param.data.numel()
            params.append(param)
    sum_params = sum_params - 2048
    print(f'{"Total":40s}\t{"":40}\t = ', sum_params)


def plot_dates_values(data):
    """
    :param data: the data to be plotted
    :return: plt of the data

    """

    dates = data['timestamp'].to_list()
    values = data["value"].to_list()
    dates = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in dates]
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation='vertical')
    ax = plt.gca()
    xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")
    ax.xaxis.set_major_formatter(xfmt)
    plt.plot(dates, values);

