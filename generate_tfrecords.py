# Add openpose to the path and import PoseEstimation
import sys
sys.path.append('./openpose')

import json
import os
import activities
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import argparse
from openpose.common import estimate_pose, draw_humans
from openpose.networks import get_network
import c3d_model

activities_list = activities.activities
samples_number = activities.samples_number

# Openpose variables
model = 'mobilenet'
input_width = 368
input_height = 368
stage_level = 6
input_node = tf.placeholder(tf.float32, shape=(1, input_height, input_width, c3d_model.CHANNELS), name = 'image')
net, _, last_layer = get_network(model, input_node, None)

# Decode data and return images and label from tfrecords
def _parse_function(serialized_example):

    # Prepare feature list; read encoded JPG images as bytes
    features = dict()
    features["class_label"] = tf.FixedLenFeature((), tf.int64)
    for i in range(c3d_model.NUM_FRAMES_PER_CLIP):
        features["frames/{:02d}".format(i)] = tf.FixedLenFeature((), tf.string)

    # Parse into tensors
    parsed_features = tf.parse_single_example(serialized_example, features)

    # Decode the encoded JPG images
    images = []
    for i in range(c3d_model.NUM_FRAMES_PER_CLIP):
        images.append(tf.image.decode_jpeg(parsed_features["frames/{:02d}".format(i)]))

    # Pack the frames into one big tensor of shape (N,H,W,3)
    images = tf.stack(images)
    label = tf.cast(parsed_features['class_label'], tf.int64)

    return images, label

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

# Create tfrecord from list of data
def create_tf_records(file_list, dest, name):

    # Create a session for running Ops on the Graph.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Load pretrained weights
    s = '%dx%d' % (input_node.shape[2], input_node.shape[1])
    ckpts = 'model/mobilenet_' + s + '/model-release'
    variables = tf.contrib.slim.get_variables_to_restore()
    loader = tf.train.Saver(variables)
    loader.restore(sess, ckpts)

    # Make the graph read-only and avoid memory leak
    # sess.graph.finalize()

    # Specify the number of files in each tfrecord
    files_per_tfrecord = len(file_list)
    number_files = int(len(file_list)/files_per_tfrecord)

    for j in range(number_files):
        print('File', j+1, '/', number_files)

        train_filename = dest + name + '_' +  str(j) + '.tfrecords'  # address to save the TFRecords file

        sub_list = file_list[j*files_per_tfrecord:(j+1)*files_per_tfrecord]

        # open the TFRecords file
        with tf.python_io.TFRecordWriter(train_filename) as writer:
            for i in tqdm(range(len(sub_list))):
                file = sub_list[i]

                # Get frames from the video
                frames = get_frames(file[1], c3d_model.NUM_FRAMES_PER_CLIP, file[2], c3d_model.CROP_SIZE, file[3], sess)
                label = activities_list[file[0]]

                # Generate poses for the frames
                poses = np.zeros(shape=(c3d_model.NUM_FRAMES_PER_CLIP, c3d_model.CROP_SIZE, c3d_model.CROP_SIZE, c3d_model.CROP_SIZE), dtype=float)

                try:
                    for z in range(len(frames)):
                        image = cv2.resize(frames[z], dsize=(input_height, input_width), interpolation=cv2.INTER_CUBIC)
                        pafMat, heatMat = sess.run(
                            [
                                net.get_output(name=last_layer.format(stage=stage_level, aux=1)),
                                net.get_output(name=last_layer.format(stage=stage_level, aux=2))
                            ], feed_dict={'image:0': [image]}
                        )
                        heatMat, pafMat = heatMat[0], pafMat[0]
                        humans = estimate_pose(heatMat, pafMat)
                        pose_image = np.zeros(tuple(image.shape), dtype=np.uint8)
                        pose_image = draw_humans(pose_image, humans)

                        # cv2.imwrite('teste.jpg', pose_image)

                        img = cv2.resize(pose_image, dsize=(c3d_model.CROP_SIZE, c3d_model.CROP_SIZE), interpolation=cv2.INTER_CUBIC)
                        poses[z, :, :, :] = img

                    # Create the dictionary with the data
                    features = {}
                    features['num_frames'] = _int64_feature(poses.shape[0])
                    features['height'] = _int64_feature(poses.shape[1])
                    features['width'] = _int64_feature(poses.shape[2])
                    features['channels'] = _int64_feature(poses.shape[3])
                    features['class_label'] = _int64_feature(label)
                    features['class_text'] = _bytes_feature(tf.compat.as_bytes(file[0]))
                    features['filename'] = _bytes_feature(tf.compat.as_bytes(file[1].split('/')[1]))

                    # Compress the frames using JPG and store in as bytes in:
                    # 'frames/01', 'frames/02', ...
                    for j in range(len(poses)):
                        ret, buffer = cv2.imencode(".jpg", poses[j])
                        features["frames/{:02d}".format(j)] = _bytes_feature(tf.compat.as_bytes(buffer.tobytes()))

                    # Wrap the data as Features
                    feature = tf.train.Features(feature=features)

                    # Create an example protocol buffer
                    example = tf.train.Example(features=feature)

                    # Serialize the data
                    serialized = example.SerializeToString()

                    # Write to the tfrecord
                    writer.write(serialized)

                except:
                    print('Error exporting frames from file!')
                    print(file)

        sys.stdout.flush()

# Wrapper for inserting int64 features into Example proto
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# Wrapper for inserting bytes features into Example proto
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Wrapper for inserting bytes features into Example proto
def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

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
    # frames = np.zeros(shape=(frames_per_step, im_size, im_size, 3), dtype=float)
    frames = []

    for z in range(frames_per_step):
        frame = start_frame + z
        video.set(1, frame)
        _, img = video.read()

        if flip:
            img = cv2.flip(img, 1)

        frames.append(img)

    return frames

# Main function
def main(json, videos, dest):

    print('\nCollecting train and test list of files')
    train_list, test_list = create_files_list(json, videos)

    print('\nAugmenting train list with', samples_number, 'samples per activity')
    train_list = augment_list(train_list)
    print('Augmented train size:', len(train_list))

    # Shuffle data
    random.shuffle(train_list)
    random.shuffle(test_list)

    # Uncomment to generate a small sample of tfrecords
    # train_list = train_list[:20]
    # test_list = test_list[:20]

    # Make test_list divisible by 100
    test_list = test_list[0:int(len(test_list)/100)*100]

    print('\nTrain size:', len(train_list))
    print('Test size: ', len(test_list))

    if not os.path.exists(dest):
        os.makedirs(dest)

    print('\nCreating tfrecords for training')
    create_tf_records(train_list, dest, 'train')

    print('\nCreating tfrecords for testing')
    create_tf_records(test_list, dest, 'test')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create tfrecords from pose videos')
    parser.add_argument('--json', dest='json', type=str, default='json/', help='path of the json files')
    parser.add_argument('--videos', dest='videos', type=str, default='videos/', help='path of the video files')
    parser.add_argument("--dest", dest="dest", type=str, default="tfrecords/", help="path to the tfrecord files")
    args = parser.parse_args()

    main(args.json, args.videos, args.dest)
