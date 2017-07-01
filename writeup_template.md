# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image9]: ./example_randon_signs.jpg "Examples"
[image10]: ./example_chart.jpg "Chart"
[normalization]: ./normalizes.jpg "Examples"
[23_examples]: ./23_examples.jpg "Chart"
[softmax_30]: ./softmax_30.jpg "Examples"
[softmax]: ./softmax.jpg "Chart"


### Data Set Summary & Exploration

I used python to explore statistics of the traffic 
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### Visualization of the dataset.

Here is some random examples of images in dataset with their label

![alt text][image9]

Here is an exploratory visualization of the data set. 
It is a bar chart showing how how each class is distributed in dataset

![alt text][image10]

### Design and Test a Model Architecture

As a first step, I decided to shuffle dataset using scikit learn library and the I define a function to normalized images. 


Normalize function pseudocode : 

for each image:
   for each pixel in image:
      pixel = (pixel - 128)/ 128

Here is an example of a traffic sign image before and after normalization.

![alt text][normalization]

Observation: the image look strange, but the values are ok,the data has mean zero and equal variance.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

### Model Architecture

I decided use the LeNet's architeture to classify the Traffic Signs dataset.

LeNet, a pioneering convolutional network by [LeCun](http://yann.lecun.com/exdb/lenet/) that classifies digits, was applied by several banks to recognise hand-written numbers on checks (cheques) digitized in 32x32 pixel images.

To use LeNet, I've ajusted:

   * first Layer to receive 32x32x2 input, the original convolutional network expected 32x32x1 images
   * last Layer to output 43 classes, the original convolutional network was seted configured to 10 classess.

My final model consisted of the following layers:

 ### Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
      strides= [1, 1, 1,1] , padding= VALID
 
      RELU Activation.
      MAX Pooling. Input = 28x28x6. Output = 14x14x6.
 
 ### Layer 2: Convolutional. Output = 10x10x16.
      strides=[1, 1, 1, 1], padding='VALID'
    
      RELU Activation.
      MAX Pooling. Input = 10x10x16. Output = 5x5x16.
      Flatten. Input = 5x5x16. Output = 400.
 
 ### Layer 3: Fully Connected. Input = 400. Output = 120.

      RELU Activation.
 
 ### Layer 4: Fully Connected. Input = 120. Output = 84.
 
      RELU Activation.
 
 ### Layer 5: Fully Connected. Input = 84. Output = 43.
 

I trained my model several times to find the optimum parameters to reach accuracy of 0.93 in validation set. I have tried using non-normalized data and gray-scaled, but the best result was using normalized 3 channels images.

In my final approach, I used [Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) with 0.001 Learning rate. My batch size was 128, number of epochs iqual 100.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.942
* test set accuracy of 0.897

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][23_examples] 

Here are the results of the prediction:

| False / True		   |     Prediction	        		      		         	| 
|:------------------:|:---------------------------------------------------:| 
| False        		| Speed limit (30km/h)   	   								| 
| False     			| Priority road				   			      			|
| True					| Speed limit (30km/h)			   							|
| True	      		| Children crossing				 	   	         		|
| True		      	| No passing      						               	|
| True         		| General caution   						          			| 
| False     			| Speed limit (20km/h)  						   			|
| True					| Stop									               		|
| True	      		| Speed limit (60km/h) 					    			   	|
| True	      		| End of no passing by vehicles over 3.5 metric tons	|
| True         		| Stop   									                  | 
| True      			| Slippery Road  							         			|
| True					| End of speed limit (80km/h)									|
| False	      		| General caution				 			                	|
| True	      		| Speed limit (120km/h)      						       	|
| True         		| Right-of-way at the next intersection   				| 
| True       			| Bumpy road            										|
| True					| Yield						               					|
| False	      		| General caution					 			             	|
| True	      		| Priority road                  							|
| True         		| Bumpy road   				            					| 
| False     			| Yield 					                  					|
| True					| Speed limit (50km/h)											|


The model was able to correctly guess 17 of the 23 traffic signs, which gives an accuracy of 73.9 %. 

This result is bellow of expected, one of reasons can be the distribution of examples, how we figure out in distribuition chart to some signs we have a lot more samples, so the model will tend to predict with more sure sign that he saw more in training.

I printed probabilities for each of these 23 images, so we can the top 5 soft max probabilities, here a example:

![alt text][softmax_30]

| Class  |         Label          |   Probability (%)                         |
|:------:|:----------------------:|:-----------------------------------------:|
|  1     | Speed limit (30km/h)   |    100                                    |
|  2     | Speed limit (50km/h)   |    0.000000000000000222                   |
|  5     | Speed limit (80km/h)   |    0.0000000000000000619                  |
|  3     | Speed limit (60km/h)   |    0.00000000000000000000000824           |
|  7     | Speed limit (100km/h)  |    0.0000000000000000000000000000000137   |

### Top 5 soft max probabilities for first 6 test images:

![alt text][softmax]

