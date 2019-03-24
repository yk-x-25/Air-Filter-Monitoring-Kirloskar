# Kirloskar
This project aims to solve one of the major problem in cars,the air filter monitoring.The problem was quoted by Kirloskar Motors at Smart India Hackathon 2019


Firstly, The dataset given was high imbalanced one. Upsampling was done to equalize the data points. Following upsampling, The dataset was split into training and testing sets .I took K nearest Neighbour for this case as a normal classifier plane won't be efficient.

KNN makes use of nearest data points so that the classifier can predict whether the label belongs to class 0 or 1 depending on the maximum
no.of points near by
