import os
import shutil
from preprocess.utils import extract_frames

path = '/y/ayhassen/anomaly_detection/ucf/final_dataset/videos/'
# base_path = '/y/ayhassen/anomaly_detection/ucf/final_dataset/train/frames'

f = open(f'/y/ayhassen/anomaly_detection/ucf/final_dataset/train.txt', 'w')  
f_val = open(f'/y/ayhassen/anomaly_detection/ucf/final_dataset/val.txt', 'w')  

count = 0

for video in os.listdir(path):
    max_frame = 0
    for frame in os.listdir(path + video): 
        if int(frame[:-4]) > max_frame:
            max_frame = int(frame[:-4])
    var = int(video[video.find('_') + 1 : video.rfind('_')])
    if var < 771:
        f.write(f'{path}{video} ' + '1' + ' ' + str(max_frame) + ' 1\n') 
    elif var > 771 and var < 964:
        f_val.write(f'{path}{video} ' + '1' + ' ' + str(max_frame) + ' 1\n') 
    elif var > 964 and var < 1004:
        f.write(f'{path}{video} ' + '1' + ' ' + str(max_frame) + ' 0\n') 
    else:
        f_val.write(f'{path}{video} ' + '1' + ' ' + str(max_frame) + ' 0\n') 
 
# for video in os.listdir(path + '/videos'):
#     if video not in os.listdir(path + '/videos'):
        # os.mkdir(f'{path}/video_folder/{video}')
#     # else:
        # print(video)
        # extract_frames(f'{path}/videos/{video}', f'{path}/video_folder/', video)  

# path = '/y/ayhassen/anomaly_detection/shanghaitech/testing_set/testing/'

# for folder in os.listdir(path + 'frames/'):
#     if folder.startswith('0'):
#         for video in os.listdir(path + 'frames/' + folder):
#                 shutil.copy2(path + 'frames/' + folder + '/' + video, path + 'merged_frames')

# path = '/y/ayhassen/anomaly_detection/ucf/final_dataset/videos/'
# for video in os.listdir(path):
#     count = 0
#     for frame in os.listdir(path + video):
#         count += 1

#     print(count, video) if count == 0 else None
#     os.rmdir(path + video) if count == 0 else None 