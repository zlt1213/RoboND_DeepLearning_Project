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
Generally speaking, the output of a convolutional layer is a 4D tensor. The main function of 1 x 1 Convolution is to flastten the 4D tensor to 2D tensor. 
### 2.4 Bilinear Upsample  

### 2.5 Transposed Convolution


## 3. Architecture of FCN  

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
## 5. Reference  
[1] http://iamaaditya.github.io/2016/03/one-by-one-convolution/  

[2] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift  



network architecture and the role that it plays in the overall network.

The student can demonstrate the benefits and/or drawbacks of different network architectures pertaining to this project and can justify the current network with factual data.


Any choice of configurable parameters should also be explained in the network architecture.


a graph, table, diagram, illustration or figure for the overall network to serve as a reference for the reviewer.


The write-up conveys the student's understanding of the parameters chosen for the the neural network.

The student explains their neural network parameters including the values selected and how these values were obtained (i.e. how was hyper tuning performed? Brute force, etc.) Hyper parameters include, but are not limited to:




The student has a clear understanding and is able to identify the use of various techniques and concepts in network layers indicated by the write-up.

The student demonstrates a clear understanding of 1 by 1 convolutions and where/when/how it should be used.

The student demonstrates a clear understanding of a fully connected layer and where/when/how it should be used.

The student has a clear understanding of image manipulation in the context of the project indicated by the write-up.

The student is able to identify the use of various reasons for encoding / decoding images, when it should be used, why it is useful, and any problems that may arise.

The student displays a solid understanding of the limitations to the neural network with the given data chosen for various follow-me scenarios which are conveyed in the write-up.

The student is able to clearly articulate whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required.
