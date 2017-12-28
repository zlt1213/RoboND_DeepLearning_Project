# RoboND Deep Learning - Follow Me Project


## 1. Introduction
![Convolutional Layer](/report/imgs/title_img.png "Title Image")  
Fully-Convolutional Neural network(FCN) is a kind of deep learning neural networks that performs well for tasks such as image identification and image segmentation. Based on the Drone Sim environment and low level control algorithms(such as PID) developed previously, the main goal of this project is to develop some higher level algorithms in order to make the drone follow a certain person in the simulation environment. FCN is implemented as the core algorithm for image identification and image segmentation.  

## 2. Architecture of FCN  
This section mainly discusses the overall architecture of the fully-convolution neural network.  

![Convolutional Layer](/report/imgs/fcn.png "Fig. 1 Overall Architecture of FCN")  

Fig. 1 Overall Architecture of FCN  

### 2.1 Structure of Encoder and Discussion  
As shown in the Fig. 2, the yellow-colored layers are separable convolutional layers. These three layers form the section of encoder in the FCN network. Each convolutional operation strides the kernel for 2 and double the depth of the layer.  

The overall structure of the encoder section is quite similar to a CNN for image identification. The encoder section helps the network to learn details of image for object classification and down-samples the input image. While focusing on the detailed portion of the inputs, the pipline of the encoder may cause the loose of some overall information.  

For the reasons above, a special technique called **skip layer** is also add. It concatenates some layers in the encoder section to the corresponding layers in the decoder section to maintain the flow of overall-scaled information.  

### 2.2 Structure of Decoder and Discussion  
In Fig.1, the decoder section of FCN is colored as green. The output of the 1 x 1 convolution is pumped into the bilinear upsample layers one after another. Finally, the output is upsampled to the same size as the input layer.  
While the encoder down-samples the image, the decoder up-samples the details thus regains some spatial information. The trained kernels for classification from the encoder are mapped to the full size as the input images.  

### 2.3 1 x 1 Convolution as Connection
The encoder section and the decoder section are connected with a 1 x 1 convolutional layer. The output of a convolutional layer is a 4D tensor. In this FCN Architecture, the 1x1 convolution helps in reducing the dimensionality of the filter space[3]. Meanwhile , the 1x1 convolution also helps to retain spatial information.  Also, 1x1 convolution is equivalent to cross-channel parametric pooling layer. “This cascaded cross channel parameteric pooling structure allows complex and learnable interactions of cross channel information”[4]. For the reasons above, the encoder and the decoder are connected by a 1 x 1 convolutional layer.

### 2.4 Overall Structure of FCN  
A typical FCN composed of a encoder section and a decoder section which are connected with a 1 x 1 convolution. The input is pumped though the whole network.  
A special technique called skip layers is introduced to improve the performs of FCN. As show in Fig. 2, some of the layers in the encoder section or even the input itself are concatenated to the decoder layers, so that the overall information in the input image can be reserved.  
In order to test and tune the performance of the network, a special parameter `num_filters` is employed to define the number of kernels in each layer.  


## 3. Types of Layers in Convolutional Neural Network  

This section mainly contains two parts. The first part of this section talks about the overall structure of the Fully-convolutional Network. The following parts explain the composition and functions of each layer one by one.  

### 3.1 Convolutional Layers  

A convolutional layer is a small neural network as show in the figure below.  The kernel(or the filter) of the convolutional layer is a 2D matrix which strides over the inputs. The elements of this matrix are called as weights. For a RGB image, there are three channels, so each kernel contains 3 matrices each of which strides on one color-channel. The main function of convolutional layer is to detect certain features in an image and pass it to the next layers in the network. The weights and biases of a kernel are shared over the whole image.  

![Convolutional Layer](/report/imgs/conv_layer.png "Fig. 2 Convolutional Layer[1]")  
Fig. 2 Convolutional Layer[1]  

### 3.2 Separable Convolutions  

A separable convolution layer mainly serves the same function as a typical convolutional layer but with less number of parameters. Convolution operation usually comes with a number of kernels. Traditionally, the convolution of an image is carried out by applying each kernel on the same RGB image for several times.  

For 9 kernels each sized 3 by 3, there are totally `9 * 3 * 3 * 3 = 243` parameters. For the separable convolution, only one kernel is applied on the input image firstly, which gives out three "feature maps". After that, each feature map is manipulated with a 1 x 1 1x1 convolution, so the total parameters for the separable convolution layer is 54(`3 * 3 * 3 + 3 * 9 = 54`).
Compared with the classical convolutional layer, the separable convolutional layer reduces the total number of parameters, which makes the training process much faster and prevents the network from over fitting.  

The separable convolution operation is defined as `SeparableConv2DKeras()` in Keras.

### 3.3 Batch Normalization  
Normalization is a common regulation method to improve the overall performance of the neural network. It is usually applied to the input of the network. On the other hand, the output of a layer can be regarded as the input to the next layer. In the light of this statement, the output may also be normalized before pumping into the next layer. Batch Normalization is such a technique that normalize the output of a layer.  

The main benefit of batch normalization are: (1) Networks train faster; (2) Allows higher learning rates; (3) Simplifies the creation of deeper networks; (4) Provides a bit of regularization  

The function `eparable_conv2d_batchnorm()` in Keras performs batch normalization.  


### 3.4 1 x 1 Convolutions and Fully Connected Layer
The structure and functionality of the 1 x 1 convolution is explained in detail in section 2.4.  

Different from 1 x 1 convolution layer, the structure and the functionality of the fully connected layer is quite different. In the **fully connected layer**, one single neural is connected to all the activations of the previously layer. The fully connected layer acts like a high-level reasoning layer in the neural network[5].  

### 3.5 Transposed Convolution  
Compared with convolutional layers, the main function of a transposed layer is to upsampling the input layer. The transposed convolution can be act as the core part in the decoder in a FCN.

### 3.6 Bilinear Upsample  
Bilinear is another method to upsample the inputs to this layer. It utilizes the weighted average of four nearest known pixels, located diagonally to a given pixel, to estimate a new pixel intensity value. The weighted average is usually distance dependent.  
It is defined as function `BilinearUpSampling2D()` in Keras.  


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

At the very beginning, the learning_rate is set to 0.01. With this learning rate, the loss stopped dropping after a few epochs. Thus, I reduced the learning_rate to a lower level. Although the network with a smaller learning_rate would take a long time to train, the loss of it would keep dropping at every epoch.  
The batch_size is determined by the size of the RAM in the GPU. I tried a very large number of batch size and ended up with a `OOM`(out of memory) error on the 1080 Ti GPU.  
During the training of certain epochs, the loss kept increasing, so I lowered the steps_per_epoch to jump out of these epochs quickly thus saving the overall training time.  

After a few test runs, the final values of hyper parameters are:
- `learning_rate = 0.0005`  
- `batch_size = 48`  
- `num_epochs = 60`  
- `steps_per_epoch = 50`  
- `validation_steps = 50`  
- `workers = 4`  

## 5. Training Process and Results
### 5.1 The Training Process
I tried to train the network on two different machines. The first one is the Amazon E2C Instance which has a Nvidia K80 inside. The second one is the my personal server(HP Z820) with Nvidia 1080 ti inside. One the E2C Instance, the average run time for a single epoch is about 510 second. Mean while, the run time for a single epoch on Nvidia 1080 ti is around 220 seconds, so I decided to train the network on my local machine.  
After a few test runs, I noticed that the overall performance of the machine will increase significantly by increasing the number of kernels in each convolutional layers. Thus I double the number of kernels and tune the batch size so that the network will fit into the GPU-RAM.  
The overall training process took more than one hour. The process and the final results can be seen in the jupyter notebook.  
The accuracy during training is show in the images below.  
![Convolutional Layer](/report/imgs/epoch_20.png "Fig. 3 Result of Target Identification")  
Fig. 3 Epoch 23/60  
![Convolutional Layer](/report/imgs/epoch_50.png "Fig. 4 Result of Target Identification")    
Fig. 4 Epoch 60/60  
### 5.2 The results  
The performance of this network structure is good enough. The trained network successfully accomplishes the tasks of image segmentation and image identification. The detailed results are contained in the project folder.  

![Convolutional Layer](/report/imgs/result_hero.png "Fig. 5 Result of Target Identification")  
Fig. 5 Result of Target Identification  

![Convolutional Layer](/report/imgs/result_with_no_tag.png "Fig. 6 Result without Target in Image")  
Fig. 6 Result without Target in Image  

![Convolutional Layer](/report/imgs/result_with_tag.png "Fig. 7 Result with Target in Image")  
Fig. 7 Result with Target in Image  

![Convolutional Layer](/report/imgs/final_score.png "Fig. 8 Final Score")  
Fig. 8 Final Score

As shown in the figures above the hero is detected successfully. The final score of the training is 0.413, which is good enough.

## 6. Discussion and Future Work
By trying different kinds of structures, it is clear that the number of kernels has a big influence on the performance of the FCN. Increasing a small amount of kernels will increase the overall performance of the FCN significantly. Limited by the RAM in the GPU, the max number of kernels of the first convolutional layer is 96. I would like to try a larger number of kernels in the future.  

Since the quality of the training process depends on the training data, so there are several techniques that can be applied to increase the quality of the data set. The first one is to increase the size of input images. A larger input size will feed more detailed information to the FCN network. Another interesting technique can be applied is to create new 'sample images' by flipping the images in the original train dataset. This may not only increase the size of the training set, by may reduce the spatial dependency as well. The last technique is quite straight forward. By collecting more images and joining them to the training set may also increase the performance quite largely.

Finally, I think that this model can be reused to detect other classes such as cats, dogs or people in white T-shirt and so one. The things need to be modified is the mask images used by the training process.  

This Follow Me project is quit interesting. I learnt a lot from this project. I will apply this tech to my self-driving cart in the very near future.  

## 7. Reference  
[1] http://iamaaditya.github.io/2016/03/one-by-one-convolution/  

[2] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift  

[3] https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network  
[4] http://iamaaditya.github.io/2016/03/one-by-one-convolution/

[5] https://en.wikipedia.org/wiki/Convolutional_neural_network
