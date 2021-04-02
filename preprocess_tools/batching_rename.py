import os 
# import glob
from pathlib import Path

def rename(data_path):
    path = Path(data_path)
    # frame_count = 0
    os.chdir(path)  
    for file in sorted(path.glob("*.json")):
        olddir = os.path.join(path, file)
        if os.path.isdir(olddir):
            continue
        filename = os.path.splitext(file)[0]
        filetype = os.path.splitext(file)[1]
        oldname = filename.split('_')
        oldname[4] = oldname[4].zfill(3)
        newname = '_'.join(oldname)
        newdir = os.path.join(path, newname + filetype)
        # print(newdir)
        os.rename(olddir, newdir)
        # frame_count += 1

path = "/home/andy/dataset备份"
activities = [
    'falling',
    'kicking',
    'smashing',
    'throwing'
]
camera_num = 3

for a in range(0, len(activities)):
    for c in range(0, camera_num):
        frame_set = "c0" + str(c + 1) + "_s01" + "_a0" + str(a + 1)
        data_path = os.path.join(path, activities[a], frame_set, 'pose')
        rename(data_path)
