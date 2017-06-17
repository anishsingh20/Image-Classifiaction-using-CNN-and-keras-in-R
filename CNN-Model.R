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

#Compiling the Model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = opt,
  metrics = "accuracy"
)

#Summary of the Model and its Architecture
summary(model)



#TRAINING PROCESS OF THE MODEL
data_augmentation <- TRUE


if(!data_augmentation) {
  model %>% fit(
    train_x,train_y ,batch_size=32,epochs=5,
    validation_data = list(test_x, test_y),
    shuffle=TRUE
  )
  
} else {
  #Generating images
  datagen <- image_data_generator(
    featurewise_center = TRUE,
    featurewise_std_normalization = TRUE,
    rotation_range = 20,
    width_shift_range = 0.30,
    height_shift_range = 0.30,
    horizontal_flip = TRUE
  )
  #Fit image data generator internal statistics to some sample data
  
  datagen %>% fit_image_data_generator(train_x)
  #Generates batches of augmented/normalized data from image data and labels
  model %>% fit_generator(
    flow_images_from_data(train_x, train_y, datagen, batch_size = 32,
                          save_to_dir="F:/PROJECTS/CNNcifarimages/"),
    steps_per_epoch=as.integer(50000/32), #no of training samples/batch size
    epochs = 5,
    validation_data = list(test_x, test_y)
    
    
  )
  
  
}

#after training
#loss: 1.5014 - acc: 0.4529 - val_loss: 2.7578 - val_acc: 0.1665 for 5 epochs( ie iterating 5 times over dataset)




#Model to yaml

yaml<-model_to_yaml(model)
class(yaml)

#saving to JSON
json<-model_to_json(model)
json

#saving the Model's architecture
save_model_hdf5(model,filepath = "F:/PROJECTS/Image-Classification", overwrite = TRUE,
                include_optimizer = TRUE)

load_model_hdf5(filepath = "F:/PROJECTS/Image-Classification",compile = T)


