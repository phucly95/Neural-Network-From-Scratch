# Build Neural Network From Scratch Without Using ML Framework(tensorflow, pytorch...)
> **_NOTE:_** This project is inspired by keras framework.
## How to run project
### To quickly test you can copy code in neural-network-standalone.js file and paste to console tab on browser
### Full Project Setup:
1. Download mnist dataset on kaggle https://www.kaggle.com/datasets/hojjatk/mnist-dataset
2. Extract and copy these files to mnist-data folder(create new one if needed):
  + t10k-images.idx3-ubyte
  + t10k-labels.idx1-ubyte
  + train-images.idx3-ubyte
  + train-labels.idx1-ubyte
3. Install dependencies first:
> npm install
4. Run project
> npm start

## Compatibility:
- node v16.14.0
- npm 8.3.1
## Example
Here is an example of how to create and summarize a model and save it after training:
> let model = new Model('categorical_crossentropy', earlyStoppingCallback, epochEndCallback, ['loss', 'progress']);
>
> model.add(new Dense(784, 10, 'leaky_relu'));
>
> model.add(new Dense(10, 10, 'softmax'));
>
> model.summary();
>
> model.train(xTrain, yTrain, 10, 0.001);
> 
> model.save('./mnist_model.any_extension');
