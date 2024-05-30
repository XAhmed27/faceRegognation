this is a documentation to run the project on kaggle and here is the data set link  https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset



[NNandCNNModels.pdf](https://github.com/XAhmed27/faceRegognation/files/15500863/NNandCNNModels.pdf)



this is a documentation to expalin how to run the SVM and use the data set


[SVMReport.pdf](https://github.com/XAhmed27/faceRegognation/files/15500867/SVMReport.pdf)


• Data exploration and preparation:
o Reshape the RGB images, so that the dimension of each image is (64,64,3)
o Convert the RGB images to greyscale.
o Normalize each image.
• Experiments and results:
o Split the data into training and testing datasets (if there is no testing dataset)
o First experiment:
▪ Train an SVM model on the grayscale images.
▪ Test the model and provide the confusion matrix and the average f-1
scores for the suitable testing dataset.

o Split the training dataset into training and validation datasets. (if there is no
validation dataset)
o Second experiment:
▪ Build 2 different Neural Networks (different number of hidden layers,
neurons, activations, etc.)
▪ Train each one of these models on the grayscale images and plot the
error and accuracy curves for the training data and validation data.
▪ Save the best model in a separated file, then reload it.

▪ Test the best model and provide the confusion matrix and the average f-
1 scores for the testing dataset.

o Third experiment:
▪ Train a Convolutional Neural Network on the grayscale images and plot
the error and accuracy curves for the training data and validation data.
▪ Train another Convolutional Neural Network on the RGB images and
plot the error and accuracy curves for the training data & validation data.
▪ Save the best model in a separated file, then reload it.

▪ Test the best model and provide the confusion matrix and the average f-
1 score fie the suitable testing dataset.

o Compare the results of the models and suggest the best model.
