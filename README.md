#  My Machine Learning Playground
Everything from 101 Linear and Logistic Regression, to Neural Networks on Google TensorFlow, to GAN's on PyTorch

[Check out this blog post](https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f) for an introduction to Generative Networks. 

[My notebooks](https://github.com/dougfoo/machineLearning/blob/master/Intro.ipynb)

## Vanilla GANs
Vanilla GANs found in this project were developed based on the original paper [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) by Goodfellow et al.

These are trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), and learn to create hand-written digit images using a 1-Dimensional vector representation for 2D input images.
- [PyTorch Notebook](https://github.com/diegoalejogm/gans/blob/master/Vanilla%20GAN%20PyTorch.ipynb)
- [TensorFlow Notebook](https://github.com/diegoalejogm/gans/blob/master/Vanilla%20GAN%20TensorFlow.ipynb)

<img src=".images/vanilla_mnist_pt_raw.png" width="300"> <img src=".images/vanilla_mnist_pt.png" width="300">

__MNIST-like generated images before & after training.__

