import os
import cv2


training_path = '/y/ayhassen/anomaly_detection/shanghaitech/training_set/videos'
frame_path = '/y/ayhassen/anomaly_detection/shanghaitech/training_set/frames'

for video in os.listdir((training_path)):
    cap = cv2.VideoCapture(os.path.join(training_path, video))

    # initialize a counter variable
    count = 0

    # loop through all the frames
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            cv2.imwrite(f'/{frame_path}/{video}_{count}.jpg', frame)
            count += 1
            print(count)
        else:
            break

    cap.release()
