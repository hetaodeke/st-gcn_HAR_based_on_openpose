import os
import numpy as np
import glob
import shutil
import argparse

def train_val_split(input_path, output_path):
    train_val_split = 0.8
    label_list = os.listdir(input_path)

    for i in range(len(label_list)):
        video_list = os.listdir(os.path.join(input_path, label_list[i]))
        for v in range(len(video_list)):
            oldpath = os.path.join(input_path, label_list[i], video_list[v])
            is_train = np.random.rand(1) < train_val_split
            if is_train:
                newpath = output_path + '/train/' + video_list[v]
                shutil.copytree(oldpath, newpath)
            else:
                newpath = output_path + '/val/' + video_list[v]
                shutil.copytree(oldpath, newpath)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Skeleton Data Converter.')
    parser.add_argument(
        '--input_path', default='/home/andy/Dataset/ucf101_skeleton/skeleton')
    parser.add_argument(
        '--output_path', default='/home/andy/Dataset/ucf101_skeleton')
    arg = parser.parse_args()
    train_val_split(arg.input_path, arg.output_path)