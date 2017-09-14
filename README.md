# NNcoloring
Image Colorization with Neural Networks

We propose a method for colorizing photos, this is, providing a color version of a given gray scale image. The method does not depend on human input, and is completely automatic. It does not depend on segmentation, scribbling or sophisticated image processing techniques. It is based on training a simple classifier using back propagation over a training set of color and corresponding gray scale pictures. The classifier predicts the color of a pixel based on the gray level of the pixels surrounding it. This small patch captures a local texture. To keep the domain for the predictor small, the colors are reduced using Self Organizing Maps. This reduction produces a small set of chroma values with enough variation as to generate good approximations for all colors in the training set.

# Authors
Matias Richart

Jorge Visca
