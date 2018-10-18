# C3D-tensorflow

This repository uses the implementation of [C3D-tensorflow][1] and [Openpose][2] to recognize body moviment from the [Human3.6M Dataset][5].

The folder `json` contains some of the videos segmentation for the training and testing set.

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

Download the two models for the training and place inside the `model` folder created. For OpenPose, save only the folder `mobilenet_368x368` and its content.

* [C3D][3]

* [OpenPose][4]

Some of the videos can be found in the following link, download them without the folders and put in the folder `videos`

* [Human3.6M Dataset][6]

Generate the dataset:
~~~~
$ python3 tf_records.py
~~~~

Train the network
~~~~
$ python3 train_c3d_ucf101.py
~~~~

[1]: https://github.com/hx173149/C3D-tensorflow
[2]: https://github.com/evalsocket/tf-openpose/tree/36fc97b2bfebb8099cb141ab96e81d925b69477b
[3]: https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0&preview=sports1m_finetuning_ucf101.model
[4]: https://drive.google.com/file/d/1RyOv5jzmMjc1EPKpF4c2TvTFoSFKU_OZ/view
[5]: http://vision.imar.ro/human3.6m/description.php
[6]: https://drive.google.com/drive/folders/1njC5FR6iGHDgF5qy_aLAFcUw1OJJSacV?usp=sharing