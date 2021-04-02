import os
import numpy as np
from pathlib import Path
import shutil

data_path = "/home/andy/dataset备份"
activities = [
    "falling",
    "kicking",
    "smashing",
    "throwing"
]
camera_num = 3
num_steps = 32
test_train_split = 0.8
overlap = 0.8125
part = ['test', 'train']


os.chdir(data_path)
for a in range(0, len(activities)):
    for c in range(0, camera_num):
        frame_set = "c0" + str(c + 1) + "_s01" + "_a0" + str(a + 1)
        olddir = os.path.join(data_path, activities[a], frame_set, "pose")
        p = Path(olddir) 
        for path in p.glob('*.json'):
            frame_id = str(path).split('_')[-2]
            filename = os.path.split(path)[1]
            is_train = np.random.rand(1) < test_train_split
            if is_train:
                newname = part[1] + '_' + activities[a] + "_c0" + str(c + 1) + '_' + frame_id + '.json'
                newdir = os.path.join(data_path, part[1], newname)
                open(newdir, 'w+')
                shutil.copyfile(path, newdir)
            else:
                newname = part[0] + '_' + activities[a] + "_c0" + str(c + 1) + '_' + frame_id + '.json'
                newdir = os.path.join(data_path, part[0], newname)
                open(newdir, 'w+')
                shutil.copyfile(path, newdir)
