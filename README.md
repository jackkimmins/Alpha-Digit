
![AlphaDigit Project Logo](https://github.com/jackkimmins/Alpha-Digit/blob/main/web/favicons/apple-icon-180x180.png)

# Alpha-Digit

This project implements a fully connected [Feedforward Neural Network (FNN)](https://en.wikipedia.org/wiki/Feedforward_neural_network) from scratch in C++ for digit classification on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) - this is a continuation from my previous attempt at this. The network is optimised for web deployment through [WebAssembly (WASM)](https://webassembly.org/), allowing it to perform inference in the browser.

## Project Details
- Feedforward Neural Network (FNN)
- Mini-Batch Gradient Descent w/ Adam Optimiser
- Early Stopping and Learning Rate Scheduling
- Multithreaded Training
- WebAssembly Module for Inferencing
- Vue3 UI
- High Classification Accuracy of **98.19%** *(on 7,000 image validation slice of the MNIST dataset)*


## Requirements
- C++20 or Later
- g++
- Emscripten (for WASM compilation)

## Demo
The demo site can be found at the following link:
[https://alpha-digit.appserver.uk/](https://alpha-digit.appserver.uk/)

## Mentions and Thanks
Special thanks to the following resources and individuals whose work greatly inspired and supported this project:

- [3Blue1Brown](https://www.youtube.com/@3blue1brown) - For the amazing video demonstrations and intuitive explanations of neural networks, can't recommend his videos enough! 😇
- Research Paper by Diederik P. Kingma and Jimmy Ba: Adam: A Method for Stochastic Optimization ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980)).
