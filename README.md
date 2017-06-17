# Image-Classifiaction #
	The 10 classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.


## Model Architecture 


#### Overview
*CIFAR-10* classification is a common benchmark problem in machine learning. The problem is to classify RGB 32x32 pixel images across 10 categories


Model's Architecture--



Layer | Description
------------ | -------------
Conv2D | A 2-D Convolution Layer with ReLu activation
Conv2D | A 2-D Convolution Layer with ReLu activation
Pool1  | Max pooling layer
Conv2D | A 2-D Convolution Layer with ReLu activation
Conv2D | A 2-D Convolution Layer with ReLu activation
Pool1  | Max pooling layer
Local1 | Fully Connected layer with ReLu activation with 512 units
Output1| Output layer with 10 Units
Softmax_Linear| Linear transformation to the outputs to compute Probabilities 
