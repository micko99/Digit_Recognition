# Recognize numbers on image

In this project, we worked on the localization and classification of digits from the image that is entered by the client. We have created the simplest html file on which we have FileField and SubmitField. Also, on the backend side, we used Flask, a micro web framework that's very suitable for small projects. Further, the backend sends the image to processing in order to localize the numbers and classificate(recognize) which numbers are in the image.

For localization, we used vertical and horizontal histogram projection.

The classification was done using CNN(convolutional neural network). We performed training model on the MNIST dataset(handwritten digits dataset) which has 60 000 images for training and 10 000 images for testing.

# How to run

1. python app.py or just on run icon (default host=localhost and port=5000).
2. You can insert just host.
    For example: (python app.py 127.0.0.1) or (python app.py host=127.0.0.1).
3. You can insert host and port.
    For example: (python app.py 0.0.0.0 5000) or you can set (python app.py host=127.0.0.1 port=5001). All combinations are allowed, but first must be HOST and after that PORT. 