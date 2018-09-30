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


def _parse_function(serialized_example):

    # Prepare feature list; read encoded JPG images as bytes
    features = dict()
    features["class_label"] = tf.FixedLenFeature((), tf.int64)
    for i in range(activities.frames_per_step):
        features["frames/{:02d}".format(i)] = tf.FixedLenFeature((), tf.string)

    # Parse into tensors
    parsed_features = tf.parse_single_example(serialized_example, features)

    # Decode the encoded JPG images
    images = []
    for i in range(activities.frames_per_step):
        images.append(tf.image.decode_jpeg(parsed_features["frames/{:02d}".format(i)]))

    # Pack the frames into one big tensor of shape (N,H,W,3)
    images = tf.stack(images)
    label = tf.cast(parsed_features['class_label'], tf.int64)

    return images, label


# Read and decode tfrecords
def read_and_decode(data_path, sess, crop_size, batch_size, gpu_num):

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path])

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Get images and label
    frames, label = decode(serialized_example, sess)

    # Reshape image data into the original shape
    frames = tf.reshape(frames, [activities.frames_per_step, crop_size, crop_size, 3])

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([frames, label], batch_size=batch_size * gpu_num, capacity=1000,
                                            min_after_dequeue=100)

    return images, labels

# Decode data and return images and label from tfrecords
def decode(serialized_example, sess):
    '''
    Given a serialized example in which the frames are stored as
    compressed JPG images 'frames/0001', 'frames/0002' etc., this
    function samples SEQ_NUM_FRAMES from the frame list, decodes them from
    JPG into a tensor and packs them to obtain a tensor of shape (N,H,W,3).
    Returns the the tuple (frames, class_label (tf.int64)
    :param serialized_example: serialized example from tf.data.TFRecordDataset
    :return: tuple: (frames (tf.uint8), class_label (tf.int64)
    '''

    # Prepare feature list; read encoded JPG images as bytes
    features = dict()
    features["class_label"] = tf.FixedLenFeature((), tf.int64)
    for i in range(activities.frames_per_step):
        features["frames/{:02d}".format(i)] = tf.FixedLenFeature((), tf.string)

    # Parse into tensors
    parsed_features = tf.parse_single_example(serialized_example, features)

    # Decode the encoded JPG images
    images = []
    for i in range(activities.frames_per_step):
        images.append(tf.image.decode_jpeg(parsed_features["frames/{:02d}".format(i)]))

    # Pack the frames into one big tensor of shape (N,H,W,3)
    images = tf.stack(images)
    label = tf.cast(parsed_features['class_label'], tf.int64)

    return images, label

# def decode(serialized_example, sess):
#   # Prepare feature list; read encoded JPG images as bytes
#   features = dict()
#   features["class_label"] = tf.FixedLenFeature((), tf.int64)
#   features["frames"] = tf.VarLenFeature(tf.string)
#   features["num_frames"] = tf.FixedLenFeature((), tf.int64)
#
#   # Parse into tensors
#   parsed_features = tf.parse_single_example(serialized_example, features)
#
#   # Define list of frames to read
#   offsets = tf.range(0, parsed_features["num_frames"] - 1)
#
#   # Decode the encoded JPG images
#   images = tf.map_fn(lambda i: tf.image.decode_jpeg(parsed_features["frames"].values[i]),
#                      offsets)
#
#   label = tf.cast(parsed_features["class_label"], tf.int64)
#
#   return images, label

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
    with tf.python_io.TFRecordWriter(train_filename) as writer:
        for i in tqdm(range(len(file_list))):
            file = file_list[i]

            # Load the image
            frames = get_frames(file[1], activities.frames_per_step, file[2], activities.im_size, file[3], sess)
            label = activities_list[file[0]]

            # Create the dictionary with the data
            features = {}
            features['num_frames'] = _int64_feature(frames.shape[0])
            features['height'] = _int64_feature(frames.shape[1])
            features['width'] = _int64_feature(frames.shape[2])
            features['channels'] = _int64_feature(frames.shape[3])
            features['class_label'] = _int64_feature(label)
            features['class_text'] = _bytes_feature(tf.compat.as_bytes(file[0]))
            features['filename'] = _bytes_feature(tf.compat.as_bytes(file[1].split('/')[1]))

            # Compress the frames using JPG and store in as bytes in:
            # 'frames/01', 'frames/02', ...
            for j in range(len(frames)):
                ret, buffer = cv2.imencode(".jpg", frames[j])
                features["frames/{:02d}".format(j)] = _bytes_feature(tf.compat.as_bytes(buffer.tobytes()))

            # Compress the frames using JPG and store in as a list of strings in 'frames'
            # encoded_frames = [tf.compat.as_bytes(cv2.imencode(".jpg", frame)[1].tobytes())
            #                   for frame in frames]
            # features['frames'] = _bytes_list_feature(encoded_frames)

            # Wrap the data as Features
            feature = tf.train.Features(feature=features)

            # Create an example protocol buffer
            example = tf.train.Example(features=feature)

            # Serialize the data
            serialized = example.SerializeToString()

            # Write to the tfrecord
            writer.write(serialized)

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
    frames = np.zeros(shape=(frames_per_step, im_size, im_size, 3), dtype=float)

    for z in range(frames_per_step):
        frame = start_frame + z
        video.set(1, frame)
        _, img = video.read()

        if flip:
            img = cv2.flip(img, 1)

        pose_frame = PoseEstimation.compute_pose_frame(img, sess)
        img = cv2.resize(img, dsize = (im_size, im_size), interpolation=cv2.INTER_CUBIC)
        frames[z, :, :, :] = img

    return frames

# Main function
def main(json, videos, dest):

    print('\nCollecting train and test list of files')
    train_list, test_list = create_files_list(json, videos)

    print('\nTrain size:', len(train_list))
    print('Test size: ', len(test_list))

    print('\nAugmenting train list with', activities.samples_number, 'samples per activity')
    train_list = augment_list(train_list)


    # train_list = train_list[:50]
    # test_list = test_list[:30]


    print('Augmented train size:', len(train_list))

    if not os.path.exists(dest):
        os.makedirs(dest)

    print('\nCreating tfrecords for train dataset')
    create_tf_records(train_list, dest, 'train')

    print('\nCreating tfrecords for validation dataset')
    create_tf_records(test_list, dest, 'val')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create tfrecords from pose videos')
    parser.add_argument('--json', dest='json', type=str, default='json/', help='path of the json files')
    parser.add_argument('--videos', dest='videos', type=str, default='videos/', help='path of the video files')
    parser.add_argument("--dest", dest="dest", type=str, default="tfrecords/", help="path to the tfrecord files")
    args = parser.parse_args()

    main(args.json, args.videos, args.dest)