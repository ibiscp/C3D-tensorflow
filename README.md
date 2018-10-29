# Human Motion Prediction

This repository uses the [C3D-tensorflow][1] and [Openpose][2] implementation to recognize body movement from the [Human3.6M Dataset][5].

The folder `json` contains some of the videos segmentation for the training and testing set.

## Commands

Clone the repository
~~~~
$ git clone --recursive https://github.com/ibiscp/C3D-tensorflow.git
~~~~

Install dependencies for Openpose
~~~
$ cd openpose
$ pip3 install -r requirements.txt
~~~

Create a folder for the pre-trained networks and one for the videos
~~~~
$ mkdir model
$ mkdir videos
~~~~

Download the two models for the training and place inside the `model` folder created. For Openpose, save only the folder `mobilenet_368x368` and its content.

* [C3D][3]

* [Openpose][4]

Some of the videos can be found in the following link, download them without the folders and put in the folder `videos`

* [Human3.6M Dataset][6]

Generate the dataset
~~~~
$ python3 generate_tfrecords.py --json=json/ --videos=videos/ --dest=tfrecords/
~~~~

Train the network
~~~~
$ python3 train.py --epochs=10 --batch_size=10 --evaluate_every=1 --use_pretrained_model=False
~~~~

### Extra file

Shows the list of activities and the frequency of activities chosen to the training
~~~~
$ python3 pose_list.py --json=json/
~~~~


## Classes

A total of 26 classes is used to train the model, these are divided in the following categories:

* Head
	* Turn right
	* Turn left
	* Raise
	* Lean forward

* Right/Left Arm
	* Shoulder extension
	* Shoulder adduction
	* Shoulder flexion
	* Shoulder abduction
	* Elbow flexion
	* Elbow extension
	* Roll the wrist

* Right/Left Leg
	* Hip flexion
	* Hip extension
	* Knee flexion
	* Knee extension

[1]: https://github.com/hx173149/C3D-tensorflow
[2]: https://github.com/evalsocket/tf-openpose/tree/36fc97b2bfebb8099cb141ab96e81d925b69477b
[3]: https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0&preview=sports1m_finetuning_ucf101.model
[4]: https://drive.google.com/file/d/1RyOv5jzmMjc1EPKpF4c2TvTFoSFKU_OZ/view
[5]: http://vision.imar.ro/human3.6m/description.php
[6]: https://drive.google.com/drive/folders/1njC5FR6iGHDgF5qy_aLAFcUw1OJJSacV?usp=sharing