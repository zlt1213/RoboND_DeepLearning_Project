# RoboND Deep Learning - Follow Me Project


## 1. Introduction

Fully-Convolutional Neural network(FCN) is a kind of deep learning neural networks that performs well for tasks such as image identification and image segmentation. Based on the Drone Sim environment and low level control algorithms(such as PID) developed previously, the main goal of this project is to develop some higher level algorithms in order to make the drone follow a certain person in the simulation environment. FCN is implemented as the core algorithm for image identification and image segmentation.  

## 2. Types of Layers in Convolutional Neural Network  

This section mainly contains two parts. The first part of this section talks about the overall structure of the Fully-convolutional Network. The following parts explain the composition and functions of each layer one by one.  

### 2.1 Convolutional Layers  

A convolutional layer is a small neural network as show in the figure below.  The kernel(or the filter) of the convolutional layer is a 2D matrix which strides over the inputs. The elements of this matrix are called as weights. For a RGB image, there are three channels, so each kernel contains 3 matrices each of which strides on one color-channel. The main function of convolutional layer is to detect certain features in an image and pass it to the next layers in the network. The weights and biases of a kernel are shared over the whole image.  

![Convolutional Layer](/report/imgs/conv_layer.png "Fig. 1 Convolutional Layer[1]")  
Fig. 1 Convolutional Layer[1]  

### 2.2 Separable Convolutions  

A separable convolution layer mainly serves the same function as a typical convolutional layer but with less number of parameters. Convolution operation usually comes with a number of kernels. Traditionally, the convolution of an image is carried out by applying each kernel on the same RGB image for several times.  

For 9 kernels each sized 3 by 3, there are totally `9 * 3 * 3 * 3 = 243` parameters. For the separable convolution, only one kernel is applied on the input image firstly, which gives out three "feature maps". After that, each feature map is manipulated with a 1 x 1 1x1 convolution, so the total parameters for the separable convolution layer is 54(`3 * 3 * 3 + 3 * 9 = 54`).
Compared with the classical convolutional layer, the separable convolutional layer reduces the total number of parameters, which makes the training process much faster and prevents the network from over fitting.  

The separable convolution operation is defined as `SeparableConv2DKeras()` in Keras.

### 2.3 Batch Normalization  
Normalization is a common regulation method to improve the overall performance of the neural network. It is usually applied to the input of the network. On the other hand, the output of a layer can be regarded as the input to the next layer. In the light of this statement, the output may also be normalized before pumping into the next layer. Batch Normalization is such a technique that normalize the output of a layer.  

The main beinifet of batch normalization are: (1) Networks train faster; (2) Allows higher learning rates; (3) Simplifies the creation of deeper networks; (4) Provides a bit of regularization  

The function `eparable_conv2d_batchnorm()` in Keras performs batch normalization.  


### 2.4 1 x 1 Convolutions  
Generally speaking, the output of a convolutional layer is a 4D tensor. The 1x1 convolution helped in reducing the dimensionality of the layer. In the FCN, the 1 x 1 convolution is also used to finish up the encoder section of the network and pass data to the decoder.  

### 2.4 Transposed Convolution  
Compared with convolutional layers, the main function of a transposed layer is to upsampling the input layer. The transposed convolution can be act as the core part in the decoder in a FCN.

### 2.5 Bilinear Upsample  
Bilinear is another method to upsample the inputs to this layer. It utilizes the weighted average of four nearest known pixels, located diagonally to a given pixel, to estimate a new pixel intensity value. The weighted average is usually distance dependent.  
It is defined as function `BilinearUpSampling2D()` in Keras.  

## 3. Architecture of FCN  
The previous section talks about the main components of a FCN one by one. This section will discuss how to put the components together in a special arrangement which will act as a fully-convolution neural network.  

![Convolutional Layer](/report/imgs/FCN.png "Fig. 2 Overall Architecture of FCN")  

Fig. 2 Overall Architecture of FCN  

### 3.1 Structure of Encoder  
As shown in the Fig. 2, the yellow-colored layers are separable convolutional layers. These three layers form the section of encoder in the FCN network. Each convolutional operation strides the kernel for 2 and double the depth of the layer.

### 3.2 Structure of Decoder  
In Fig.2, the decoder section of FCN is colored as green. The output of 1 x 1 convolution is pumped into the bilinear upsample functions one after another. Finally, the output is upsampled to the same size as the input layer.

### 3.3 1 x 1 Convolution as Connection
The encoder section and the decoder section are connected with a 1 x 1 convolutional layer. The main function of 1 x 1 convolution is explained in section 2.4.

### 3.3 Overall Structure of FCN  
A typical FCN composed of a encoder section and a decoder section which are connected with a 1 x 1 convolution. The input is pumped though the whole network.  
A special technique called skip layers is introduced to improve the performs of FCN. As show in Fig. 2, some of the layers in the encoder section or even the input itself are concatenated to the decoder layers, so that the overall information in the input image can be reserved.  
In order to test and tune the performance of the network, a special parameter `num_filters` is employed to define the number of kernels in each layer.  

## 4. Implementation of Follow Me Project  

### 4.1 Python Code, Tensorflow and Keras
The program in written in Python. Keras is a high level deep learning API based on Tensorflow. With Keras, it's mush easier to build the high level architecture of neural networks.  The main Keras API functions used in the program are:
  - `SeparableConv2DKeras()`        separable convolution  
  - `BatchNormalization()`          batch normalization of a layer  
  - `Conv2D()`                      classical convolution  
  - `BilinearUpSampling2D()`        bilinear up sampling of a input  

With the help of these functions, the encoder block and decoder block are constructed. One encoder block contains a operation of separable convolution followed by a operation of batch normalization. One decoder block contains one upsample operation, the result of which is concatenated with a `large_ip_layer`. After that, the output is pumped into two stages of separable convolutions.  

The encoder and the decoder and connected with a 1x1 convolution implemented by `conv2d_batchnorm()` with kernel size equals to 1.  

### 4.2 Data Collection

The training data and validation data are collected from QuadSim software provided by Udacity. The simulator can save image, depth map and person-mask in real time to a specified folder. In this project, the collected data is saved at `\data\raw_sim_data\train(validation)\run*`(such as run1 or run2 for different runs).  
In order to capture as much as images with the hero at the center, the first data collection run is designed in such a way that the hero is flowing a straight line while the drone is zig-zag about the line.  
Instead of using directly, the data collected is preprocessed with the help of `preprocess_ims.py`. The preprocess transform the depth map to the binary mask for the training of the FCN. On the other hand, the preprocess reduces the size of the raw data, thus makes the raw data suitable for uploading to AWS.

### 4.3 Hyper Parameters  
For this FCN, there are mainly 5 hyper-parameters to tune. They are:
- **learning_rate**: defines the ratio that how much the weights are updated in each propagation.
- **batch_size**: number of training samples/images that get propagated through the network in a single pass.
- **num_epochs**: number of times the entire training dataset gets propagated through the network.
- **steps_per_epoch**: number of batches of training images that go through the network in 1 epoch. We have provided you with a default value. One recommended value to try would be based on the total number of images in training dataset divided by the batch_size.
- **validation_steps**: number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset. We have provided you with a default value for this as well.
- **workers**: maximum number of processes to spin up. This can affect your training speed and is dependent on your hardware. We have provided a recommended value to work with.  

Among these hyper parameters, three of them are critical to the training process of the FCN. They are **learning_rate**, **batch_size** and **num_epochs**. The **learning_rate** determines how much will the weights be updated in each propagation pass. Although a higher learning rate may fasten the training process, it may also make the results less accurate. The determination of batch size is based on the hardware ability of the GPU. Number of epochs is the number of loops the training process will perform. After trying several different settings, the final hyper paramaters are determined as in the jupyter notebook.

## 6. Training Process and Results
### 6.1 The Training Process
I tried to train the network on two different machines. The first one is the Amazon E2C Instance which has a Nvidia K80 inside. The second one is the my personal server(HP Z820) with Nvidia 1080 ti inside. One the E2C Instance, the average run time for a single epoch is about 510 second. Mean while, the run time for a single epoch on Nvidia 1080 ti is around 220 seconds, so I decided to train the network on my local machine.  
After a few test runs, I noticed that the overall performance of the machine will increase significantly by increasing the number of kernels in each convolutional layers. Thus I double the number of kernels and tune the batch size so that the network will fit into the GPU-RAM.  
The overall training process took more than one hour. The process and the final results can be seen in the jupyter notebook.  
### 6.2 The results  


## 7. Discussion and Future Work

## 8. Reference  
[1] http://iamaaditya.github.io/2016/03/one-by-one-convolution/  

[2] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift  
