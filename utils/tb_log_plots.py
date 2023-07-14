from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import numpy as np






# dfw = experiment.get_scalars(pivot=False)
# print(dfw.head())
#
# Filter the DataFrame to only validation data, which is what the subsequent
# analyses and visualization will be focused on.
# Get the optimizer value for each row of the validation DataFrame.
# hue = df_run_1_nll_test.run.apply(lambda run: run.split("2")[0])
#
# plt.figure(figsize=(16, 6))
# plt.subplot(1, 2, 1)
# sns.lineplot(data=df_run_1_nll_test, x="step", y="value",
#              hue=hue).set_title("accuracy")
# plt.subplot(1, 2, 2)
# sns.lineplot(data=df_run_1_nll_test, x="step", y="value",
#              hue=hue).set_title("loss")
# plt.show()
# #
# # Perform pairwise comparisons between the minimum validation losses
# # from the three optimizers.
# _, p_adam_vs_rmsprop = stats.ttest_ind(
#     adam_min_val_loss["epoch_loss"],
#     rmsprop_min_val_loss["epoch_loss"])
# _, p_adam_vs_sgd = stats.ttest_ind(
#     adam_min_val_loss["epoch_loss"],
#     sgd_min_val_loss["epoch_loss"])
# _, p_rmsprop_vs_sgd = stats.ttest_ind(
#     rmsprop_min_val_loss["epoch_loss"],
#     sgd_min_val_loss["epoch_loss"])
# print("adam vs. rmsprop: p = %.4f" % p_adam_vs_rmsprop)
# print("adam vs. sgd: p = %.4f" % p_adam_vs_sgd)
# print("rmsprop vs. sgd: p = %.4f" % p_rmsprop_vs_sgd)
def find_stats_experiments(df,outfilename):
    unique_exp_list = df.run.apply(lambda run: run.split("2")[0]).unique()
    for exp_name in unique_exp_list:
        outfile = open(outfilename, "a")
        outfile.write('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>' + exp_name + '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>' )
        outfile.write('\n')
        outfile.close()
        df_exp = df[df.run.str.startswith(exp_name)]
        unique_tag_list = df_exp['tag'].unique()
        for tag_name in unique_tag_list:
            outfile = open(outfilename, "a")
            outfile.write(tag_name)
            outfile.write('\n')
            outfile.close()
            min_list = []
            #extract a particular tag
            df_tag = df_exp[df_exp.tag.str.startswith(tag_name)]
            #groupy different runs for that tag
            dfs = dict(tuple(df_tag.groupby('run')))
            for key, value in dfs.items():
                df_local = value
                values = np.array(df_local['value'])
                if len(values) > 470:
                    min_list.append(np.min(values))
            outfile = open(outfilename, "a")
            outfile.write('mean: ' + str(np.mean(min_list)) + ' std: ' + str(np.std(min_list)))
            outfile.write('\n')
            outfile.close()

def plot_curves_seaborn(df):
    unique_tag_list = df['tag'].unique()

    for i,tag_name in enumerate(unique_tag_list):
        df_tag = df[df.tag.str.startswith(tag_name)]
        hue = df_tag.run.apply(lambda run: run.split("2")[0])

        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df_tag, x="step", y="value",
                     hue=hue).set_title(tag_name)
        plt.show()



#### TODO: Both functions not working. Try to do





if __name__ == "__main__":
    experiment_id = "NXJSaeljSjGpikdJMxpuqQ"
    experiment_name = 'conditioning_v3'
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    outfile = '/home/vshaj/CLAS/meta_dynamics/experiments/pam/runs/results/' + experiment_name +'.txt'
    df = experiment.get_scalars()
    find_stats_experiments(df,outfile)
    #plot_curves_seaborn(df)
