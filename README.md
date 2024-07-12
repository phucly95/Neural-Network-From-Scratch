# Build Neural Network From Scratch Without Using ML Framework(tensorflow, pytorch...)
> **_NOTE:_** This project is inspired by keras framework.
## How to run project
- Download mnist dataset on kaggle https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- Extract and copy these files to mnist-data folder(create new one if needed):
  + t10k-images.idx3-ubyte
  + t10k-labels.idx1-ubyte
  + train-images.idx3-ubyte
  + train-labels.idx1-ubyte
- Install dependencies first:
> npm install
- Run project
> npm start

## This project currently work on:
- node v16.14.0
- npm 8.3.1
## Example
> let model = new Model('categorical_crossentropy', earlyStoppingCallback, epochEndCallback, ['loss', 'progress']);
> model.add(new Dense(784, 10, 'leaky_relu'));
> model.add(new Dense(10, 10, 'softmax'));
> model.summary();
