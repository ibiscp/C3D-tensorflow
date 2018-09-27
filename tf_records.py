import json
import os
import activities
import cv2
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import argparse
# Add openpose to the path and import PoseEstimation
sys.path.append('./openpose')
import PoseEstimation

activities_list = activities.activities_tfrecords
samples_number = activities.samples_number

# Create list of videos for training and testing
def create_files_list(json_dir, video_path):

    files = os.listdir(json_dir)

    train_list = []
    test_list = []

    for f in files:
        with open(json_dir + f) as file:
            Json_dict = json.load(file)

            for video in list(Json_dict.keys()):
                for activity in list(Json_dict[video]):
                    if (activity['label'] in activities_list):
                        segment = activity['milliseconds']
                        if 'train' in f:
                            train_list.append([activity['label'], video_path + video, segment, False])
                        else:
                            test_list.append([activity['label'], video_path + video, segment, False])

    return train_list, test_list

# Given a list of videos, augment in order to have n samples in each category
def augment_list(list):

    final_list = []

    for a in activities_list:
        videos = [v for v in list if v[0] == a]
        oposite_video = []

        activity = a
        if (activity[0] == 'r'):
            activity = 'l' + activity[1:]
        elif (activity[0] == 'l'):
            activity = 'r' + activity[1:]

        if (activity[0] == 'r' or activity[0] == 'l'):
            oposite_video = [[a,v[1],v[2],True] for v in list if v[0] == activity]

        augmented_list = videos + oposite_video

        # Extract n samples from each one
        augmented_list = random.sample(augmented_list, min(samples_number, len(augmented_list)))
        while len(augmented_list) < samples_number:
            samples = min(samples_number - len(augmented_list), len(augmented_list))
            augmented_list = augmented_list + random.sample(augmented_list, samples)

        final_list = final_list + augmented_list

    return final_list

def create_tf_records(file_list, dest, name):

    train_filename = dest + name + '.tfrecords'  # address to save the TFRecords file

    # Create a session for running Ops on the Graph.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)

    for i in tqdm(range(len(file_list))):
        file = file_list[i]

        # Load the image
        img = get_frames(file[1], activities.frames_per_step, file[2], activities.im_size, file[3], sess)
        label = activities_list[file[0]]

        # Create a feature
        feature = {name + '/label': _int64_feature(label),
                   name + '/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

# Convert data for tfrecords
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Extract frames from the videos
def get_frames(video_path, frames_per_step, segment, im_size, flip, sess):

    # Load video and acquire its parameters usingopencv
    video = cv2.VideoCapture(video_path)
    fps = (video.get(cv2.CAP_PROP_FPS))
    video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    max_len = video.get(cv2.CAP_PROP_POS_MSEC)

    # Check segment consistency
    if (max_len < segment[1]):
        segment[1] = max_len

    # Define start frame
    central_frame = (np.linspace(segment[0], segment[1], num=3)) / 1000 * fps
    start_frame = central_frame[1] - frames_per_step / 2

    # Matrix for the frames
    frames = np.zeros(shape=(frames_per_step, im_size, im_size, 3), dtype=float)

    for z in range(frames_per_step):
        frame = start_frame + z
        video.set(1, frame)
        ret, im = video.read()

        if flip:
            im = cv2.flip(im, 1)

        pose_frame = PoseEstimation.compute_pose_frame(im, sess)
        res = cv2.resize(pose_frame, dsize = (im_size, im_size),
                         interpolation=cv2.INTER_CUBIC)
        frames[z, :, :, :] = res

    return frames

# Main function
def main(json, videos, dest):

    print('\nCollecting train and test list of files')
    train_list, test_list = create_files_list(json, videos)

    print('\nTrain size:', len(train_list))
    print('Test size: ', len(test_list))

    print('\nAugmenting train list with', activities.samples_number, 'samples per activity')
    train_list = augment_list(train_list)

    print('Augmented train size:', len(train_list))

    if not os.path.exists(dest):
        os.makedirs(dest)

    print('\nCreating tfrecords for train dataset')
    create_tf_records(train_list, dest, 'train')

    print('\nCreating tfrecords for test dataset')
    create_tf_records(test_list, dest, 'test')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create tfrecords from pose videos')
    parser.add_argument('--json', dest='json', type=str, default='json/', help='path of the json files')
    parser.add_argument('--videos', dest='videos', type=str, default='videos/', help='path of the video files')
    parser.add_argument("--dest", dest="dest", type=str, default="tfrecords/", help="path to the tfrecord files")
    args = parser.parse_args()

    main(args.json, args.videos, args.dest)