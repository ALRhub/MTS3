import os
from matplotlib import pyplot as plt

def plot_functions(target_x, target_y, context_x, context_y, pred_y, var, dimList=[0], exp_name='trial', folder_path=None, download=True):
    """Plots the predicted mean and variance and the context points.

    Args:
      target_x: An array of shape batchsize x number_targets x 1 that contains the
          x values of the target points.
      target_y: An array of shape batchsize x number_targets x 1 that contains the
          y values of the target points.
      context_x: An array of shape batchsize x number_context x 1 that contains
          the x values of the context points.
      context_y: An array of shape batchsize x number_context x 1 that contains
          the y values of the context points.
      pred_y: An array of shape batchsize x number_targets x 1  that contains the
          predicted means of the y values at the target points in target_x.
      pred_y: An array of shape batchsize x number_targets x 1  that contains the
          predicted variance of the y values at the target points in target_x.
    """
    if folder_path is None:
        raise ValueError('Give a Folder Name / Path')
    for dim in dimList:# Plot everything(or just first memeber of batch?)
        plt.plot(target_x[0], pred_y[0], "b", linewidth=2)
        plt.plot(target_x[0], target_y[0], "k:", linewidth=2)
        plt.plot(context_x[0], context_y[0], "ko", markersize=10)
        plt.fill_between(
            target_x[0, :, dim],
            pred_y[0, :, dim] - var[0, :, dim],
            pred_y[0, :, dim] + var[0, :, dim],
            alpha=0.2,
            facecolor="#65c9f7",
            interpolate=True,
        )

        # Make the plot pretty
        plt.yticks([-2, 0, 2], fontsize=16)
        plt.xticks([-2, 0, 2], fontsize=16)
        plt.ylim([-2, 2])
        plt.grid(False)
        # ax = plt.gca()
        # ax.set_axis_bgcolor('white')
        if download:
            if os.path.exists(folder_path):
                pass
            else:
                os.mkdir(folder_path)
            plt.savefig(folder_path + "/plot_dim_" + str(dim) + "_" + exp_name + ".png")
        else:
            plt.show()