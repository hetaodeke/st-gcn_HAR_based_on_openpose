import os
import json
# from pathlib import Path
import glob
import argparse

part = ['train', 'val']
label_list = os.listdir("data/ucf101_skeleton/skeleton")

def create_label_json(input_dir, output_dir):
    for p in part:
        label_dict = {}
        video_list = os.listdir(os.path.join(input_dir, p))
        for video in video_list:
            video_name = video.split('.')[0]
            label = video_name.split('_')[1]
            label_index = label_list.index(label)
            # print(label_index)
            sample_info = {"label_index":label_index}
            label_dict[video_name] = sample_info
        output_file = p + '_label.json'
        output_path = os.path.join(output_dir, output_file)
        with open(output_path, 'w+') as f:
            json.dump(label_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Dataset Label.')
    parser.add_argument(
        '--data_path', default='/home/andy/Dataset/ucf101_skeleton')
    parser.add_argument(
        '--out_folder', default='/home/andy/Dataset/ucf101_skeleton')
    arg = parser.parse_args()
    create_label_json(arg.data_path, arg.out_folder)


