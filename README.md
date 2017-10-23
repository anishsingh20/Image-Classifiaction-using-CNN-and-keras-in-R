# Image-Classifiaction 
   I have build a dense CNN network for classifying the images in R into 10 classes. 
    
    The 10 classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
    The usual method for training a network to perform N-way classification is multinomial logistic regression, aka. softmax regression. 
    
    __Softmax__ activation is used at the end which applies a softmax nonlinearity to the output of the network and calculates the __cross-entropy__ loss between the probabilities calculated using the softmax activation function. For regularization, we also apply the usual weight decay losses to all learned variables.
    The softmax function is a generalization of __logistic(sigmoid)__ function which is used to compute the probabilites of the real valued predictions at the end of fully connected dense layer and then use those probability values to compute loss(error) values. Below is the softmax activation function.
    
        ![github logo](https://wikimedia.org/api/rest_v1/media/math/render/svg/46c32a5089726d673c30a0abfda7b35ecf0fe3ca)
    
    The objective function for the model is the sum of the cross entropy loss and all these weight decay terms, as 
    returned by the loss() function.

## Dependencies  
   __devtools::install_github('rstudio/keras')__

 
   __library(reticulate) #interface for Python in R__
 
   __library(tensorflow) #For all computations running in the backend on CPU__

   __library(keras)__
 
 
 ## Model Architecture 


#### Overview
*CIFAR-10* classification is a common benchmark problem in machine learning. The problem is to classify RGB 32x32 pixel images across 10 categories


Model's Architecture--



Layer | Description
------------ | -------------
 Conv2D-1 | A 2-D Convolution Layer with ReLu activation
Conv2D-1 | A 2-D Convolution Layer with ReLu activation
Pool-1  |  Max pooling layer
Conv2D-2 | A 2-D Convolution Layer with ReLu activation
Conv2D-2 | A 2-D Convolution Layer with ReLu activation
Pool-2  |  Max pooling layer
Local-1 |  Fully Connected layer with ReLu activation and 512 units
Output-1|  Output layer with 10 Units(each for a class)
Softmax_activation| Non-Linear transformation to the outputs to compute Probabilities 




# Model's Output

![GitHub Logo](https://thkimorgblog.files.wordpress.com/2016/03/e18489e185b3e1848fe185b3e18485e185b5e186abe18489e185a3e186ba-2016-03-12-e1848be185a9e1848ce185a5e186ab-1-02-16.png?w=764)





### Plot of Epochs(no of iterations over the Training dataset) vs Accuracy Of the Model 


![GitHub Logo](http://imagine.enpc.fr/~zagoruys/cifar.png)

