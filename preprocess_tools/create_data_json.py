import os
import json
from pathlib import Path
import glob
import argparse


def create_json(input_path, output_path):
    label_list = os.listdir(input_path)
    for i in range(len(label_list)):
        data_info = {"label":label_list[i], "label_index":i}
        video_list = os.listdir(os.path.join(input_path, label_list[i]))
        for j in range(len(video_list)):
            video_index = str(j).rjust(3, '0')
            video_info = {"video_index":'v'+video_index}
            if not os.path.exists(os.path.join(output_path, label_list[i], video_list[j])):
                os.makedirs(os.path.join(output_path, label_list[i], video_list[j]))
            frame_list = os.listdir(os.path.join(input_path, label_list[i], video_list[j]))
            os.chdir(os.path.join(input_path, label_list[i], video_list[j]))
            for f in range(len(frame_list)):
                sequence_info = {"skeleton":[]}
                frame_index = str(f).rjust(3, '0')
                sequence_info["frame_id"] = frame_index
                with open(frame_list[f], 'r+') as file:
                    sample_info = json.load(file)
                    for pose_keypoints in sample_info['people']:
                        pose = []
                        score = []
                        skeleton_info = {}
                        for idx, items in enumerate(pose_keypoints['pose_keypoints_2d']):
                            if (idx + 1)%3 == 0:
                                score.append(items)
                            else:
                                pose.append(items)
                        skeleton_info["pose"] = pose
                        skeleton_info["score"] = score
                        sequence_info["skeleton"] += [skeleton_info]
                video_info["data"] = sequence_info
                data_info["video"] = video_info
                newjson_name = label_list[i] + '_v' + video_index + '_f' + frame_index + '.json'
                new_json = os.path.join(output_path, label_list[i], video_list[j], newjson_name)
                with open(new_json, 'w+') as out_file:
                    json.dump(data_info,out_file)                




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='/home/andy/Dataset/ucf101_skeleton/output')
    parser.add_argument(
        '--out_folder', default='/home/andy/Dataset/ucf101_skeleton/skeleton')
    arg = parser.parse_args()
    create_json(arg.data_path, arg.out_folder)

