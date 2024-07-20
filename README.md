Image Classification with Convolutional Neural Network (CNN)

This project demonstrates an image classification model built using Convolutional Neural Networks (CNNs) in TensorFlow/Keras. The model is trained on the CIFAR-100 dataset and can predict the class of new images provided by the user.
Project Structure

This project consists of two main parts:

    Model Training
    Model Deployment for Predictions

Model Training

The training script covers the entire pipeline from loading the dataset to training the model and saving it for later use.
Key Steps:

    Importing Necessary Libraries:
        TensorFlow/Keras for building and training the model.
        NumPy for data manipulation.
        Matplotlib for data visualization.
        Scikit-learn for splitting the dataset.

    Loading and Exploring the CIFAR-100 Dataset:
        The CIFAR-100 dataset is loaded, containing 100 classes with 600 images each.
        The training set consists of 50,000 images and the test set contains 10,000 images.
        The dataset is preprocessed by normalizing the pixel values to the range [0, 1].

    Visualizing the Data:
        Display the shapes of the training and test datasets.
        Visualize sample images with their corresponding labels.

    Defining the Model Architecture:
        A Sequential CNN model is defined with several layers:
            Convolutional layers with ReLU activation.
            MaxPooling layers.
            Flatten layer to convert the 2D matrix to a vector.
            Dense layers with ReLU and softmax activation functions.

    Compiling and Training the Model:
        The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss.
        The model is trained for 1000 epochs with a batch size of 32.
        The training process includes validation on the test set.

    Evaluating the Model:
        Evaluate the trained model on the test dataset.
        Print the test accuracy.
        Make predictions on the test set and visualize the results.

    Saving the Model:
        Save the trained model to a file for later use.

Model Deployment for Predictions

The deployment script allows you to use the trained model to make predictions on new images provided by the user.
Key Steps:

    Importing Necessary Libraries:
        TensorFlow/Keras for loading the trained model.
        NumPy for data manipulation.
        Matplotlib for data visualization.
        Pillow for image processing.

    Loading the Trained Model:
        Load the saved model from the file.

    Defining Class Names:
        Define an array of class names corresponding to the 100 classes in CIFAR-100.

    Preprocessing New Images:
        A function to preprocess new images:
            Load the image and convert it to RGB mode.
            Resize the image to 32x32 pixels.
            Normalize the pixel values to the range [0, 1].
            Expand the dimensions to match the model's input shape.

    Making Predictions:
        Make predictions on the preprocessed image using the loaded model.
        Convert the prediction to a human-readable class name.

    Visualizing Predictions:
        Plot the image with its predicted label and confidence score.
        Visualize the prediction probabilities as a horizontal bar plot with class names on the y-axis.

How to Use

    Prepare the Environment:
        Ensure you have Python and necessary libraries installed:
            TensorFlow
            Keras
            Pillow
            Matplotlib
            NumPy

    Run Model Training:
        Execute the training script to train the model on the CIFAR-100 dataset.
        The trained model will be saved as model.keras.

    Run Model Deployment:
        Execute the deployment script.
        Provide the path to the image when prompted.
        The script will preprocess the image, make a prediction using the loaded model, and display the results.

Requirements

    Python 3.x
    TensorFlow
    Keras
    Pillow
    Matplotlib
    NumPy

Acknowledgments

This project was developed to demonstrate the application of CNNs for image classification tasks using the CIFAR-100 dataset. The example model used here can be adapted and trained on different datasets to suit various classification needs.
