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

### 3.2 Structure of Decoder  

### 3.3 Overall Structure of FCN  

## 4. Implementation of Follow Me Project  

### 4.1 Data Collection  

### 4.2 Hyper Parameters  

Epoch  
Learning Rate  
Batch Size  
Etc.  
## 6. Reference  
[1] http://iamaaditya.github.io/2016/03/one-by-one-convolution/  

[2] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift  
