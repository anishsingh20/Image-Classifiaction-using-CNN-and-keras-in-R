#Image Classification on Cifar 10


library(keras)

cifar<-dataset_cifar10()


#Training Data
train_x<-cifar$train$x / 255
train_y<-to_categorical(cifar$train$y,num_classes = 10)

#Test Data
test_x<-cifar$test$x/255
test_y<-to_categorical(cifar$test$y,num_classes = 10)


#checking the dimentions
dim(train_x)

cat("No of training samples\t--",dim(train_x)[[1]] ,
    "\tNo of test samples\t--",dim(test_x)[[1]])


#Defining the Model

model<-keras_model_sequential()


#Configuring the Model
model %>%
  layer_conv_2d(filter=48,kernel_size=c(3,3),padding="same",
                input_shape=c(32,32,3)) %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter=48,kernel_size=c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  layer_conv_2d(filter=48 , kernel_size=c(3,3),padding="same") %>%
  layer_activation("relu") %>%
  layer_conv_2d(filter=48,kernel_size=c(3,3) ) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  #flatten the input
  layer_flatten() %>%
  layer_dense(512) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  #output layer-10 classes-10 units
  layer_dense(10) %>%
  #applying softmax nonlinear activation function to the output layer to calculate
  #cross-entropy
  layer_activation("softmax") #for computing Probabilities of classes-"logit(log probabilities)


#Optimizer -rmsProp to do parameter updates 
opt <- optimizer_rmsprop(lr = 0.0001, decay = 1e-6)


model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)

