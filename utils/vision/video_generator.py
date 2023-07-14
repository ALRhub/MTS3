import cv2
import numpy
import numpy as np
import torch
import wandb

def create_video_output(image_series, path = "output.mp4", duration = 3, wandb_run = None, wandb_key =""):
    """
    Creates video from a 3d numpy array and saves it as mp4 video under the specified path
    Args:
        image_series: video to play with shape (length_timeseries, resolution_x, resolution_y
        path: path to save the video to
        duration: Length of the video (used to determine fps)

    """
    if isinstance(image_series, torch.Tensor):
        image_series = np.array(image_series.cpu())
    length, size_x, size_y = image_series.shape
    size = size_x, size_y
    fps =  int(length / duration)

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (size[1], size[0]), False)
    for i in range(fps*duration):

        data = image_series[i,:, :]
        data = data
        data = data.astype("uint8")
        out.write(data)
    out.release()
    if wandb_run is not None:
        key = f"{wandb_key}_" + str(wandb_run.step)
        wandb_run.log({key: wandb.Video(path, format="mp4")})
        #os.remove(folder_name + "/pca_" + exp_name + ".png")


if __name__ == '__main__':

    pred = torch.load("experiments/saved_predictions/out_mean0.pt")
    pred = np.array(pred)
    create_video_output(pred[0, :], path="experiments/saved_videos/out_mean0.mp4")

    #obs = torch.load("experiments/saved_predictions/tar_tar_batch100.pt")
    #obs = np.array(obs)
    #create_video_output(obs[0, :], path="tar_tar_batch100.mp4")