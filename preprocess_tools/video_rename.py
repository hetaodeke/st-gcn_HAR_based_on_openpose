import os

data_path = 'data/ucf101_skeleton/skeleton/HandstandPushups'
video_list = sorted(os.listdir(data_path))
for i in range(len(video_list)):
    old_path = os.path.join(data_path, video_list[i])
    print(old_path)
    name_list = video_list[i].split('_')
    name_list[1] = 'HandstandPushups'
    new_name = '_'.join(name_list)
    # print(new_name)
    new_path = os.path.join(data_path, new_name)
    os.rename(old_path, new_path)
    
