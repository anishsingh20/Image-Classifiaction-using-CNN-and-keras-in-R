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
  layer_conv_2d()
