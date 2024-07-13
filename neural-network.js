import fs from 'fs';
import cliProgress from 'cli-progress';
import colors from 'ansi-colors';

const mappedClasses = new Map();
const initHE = ['relu', 'leaky_relu', 'softmax'];

// sum all items in array 2d
export function sumArr(arr) {
    return arr.reduce((acc, curr) => acc + curr, 0);
}
// random in range [min,max]
export function getRandomArbitrary(min, max) {
    return Math.random() * (max - min) + min;
}

export function randn() {
    // Create random in range [0, 1)
    let u1 = Math.random();
    let u2 = Math.random();

    // Box-Muller Transform
    let z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return z0;
}

/* weights look like:
[[w00,w01,w02],
 [w10,w11,w12]]

transpose to
[[w00,w10],
 [w01,w11],
 [w02,w12]]
*/
export function transpose(weights) {
    let result = []
    for (let i = 0; i < weights[0].length; i++) {
        let r = [];
        for (let j = 0; j < weights.length; j++) {
            r.push(weights[j][i])
        }
        result.push(r);
    }
    return result;
}

export function heInit(fanIn, fanOut) {
    const scale = Math.sqrt(2 / fanIn);
    scale * randn();
    let result = [];
    for (let i = 0; i < fanIn; i++) {
        let wr = [];
        for (let j = 0; j < fanOut; j++) {
            wr.push(scale * randn())
        }
        result.push(wr);
    }
    return result;
}

export function xavierInit(fanIn, fanOut) {
    const scale = Math.sqrt(6 / (fanIn + fanOut));
    let result = [];
    for (let i = 0; i < fanIn; i++) {
        let wr = [];
        for (let j = 0; j < fanOut; j++) {
            wr.push(scale * randn())
        }
        result.push(wr);
    }
    return result;
}

// sigmoid
export function sigmoid(arr) {
    return arr.map(x => 1.0 / (1 + Math.exp(-x)));
}

export function sigmoidDerivative(arr) {
    return arr.map(x => x * (1 - x));
}
// relu
export function relu(arr) {
    return arr.map(x => x > 0 ? x : 0);
}
export function reluDerivative(arr) {
    return arr.map(x > 0 ? 1 : 0);
}

export function leakyRelu(arr, alpha = 0.001) {
    return arr.map(x => x > 0 ? x : alpha * x);
}
export function leakyReluDerivative(arr, alpha = 0.001) {
    return arr.map(x => x > 0 ? 1 : alpha);
}
// softmax
export function softmax(arr) {
    let maxX = Math.max(...arr);
    let expX = arr.map((v) => Math.exp(v - maxX));
    let sum = expX.reduce((acc, curr) => acc + curr, 0);
    return expX.map((v) => v / sum);
}
export function softmaxDerivative(arr) {
    let s = softmax(arr);
    return arr.map((v, idx) => s[idx] * (1 - s[idx]));
}
// MSE loss
export function mse(y_pred, y_true) {
    return y_pred.map((_, idx) => (y_pred[idx] - y_true[idx]) ** 2);
}
export function mseGradient(y_pred, y_true) {
    return y_pred.map((_, idx) => (y_pred[idx] - y_true[idx]));
}
// Categorical Cross Entropy loss
export function categoricalCrossEntropy(y_pred, y_true) {
    return y_true.map((_, i) => (-y_true[i] * Math.log(y_pred[i])));
}

export function categoricalCrossEntropyGradient(y_pred, y_true) {
    return y_true.map((_, i) => (y_pred[i] - y_true[i]));
}

export function binaryCrossEntropy(y_pred, y_true) {
    let loss = [];
    for (let i = 0; i < y_true.length; i++) {
        let currentLoss = y_true[i] === 1 ? -Math.log(y_pred[i]) : -Math.log(1 - y_pred[i]);
        loss.push(currentLoss);
    }
    return loss;
}
export function binaryCrossEntropyGradient(y_pred, y_true) {
    let gradients = [];
    for (let i = 0; i < y_true.length; i++) {
        if (y_true[i] === 1) {
            gradients.push(-1.0 / y_pred[i]);
        } else {
            gradients.push(1.0 / (1 - y_pred[i]));
        }
    }
    return gradients;
}

export function writeFile(path, data) {
    fs.writeFileSync(path, data, 'utf8');
}

export function readFile(path) {
    let data
    try {
        data = fs.readFileSync(path, 'utf8');
    } catch (err) {
        console.error(err);
    }
    return data
}

export function convertToOneHot(labels, numClasses) {
    const oneHotVectors = [];

    for (const label of labels) {
        const oneHotVector = new Array(numClasses).fill(0);
        oneHotVector[label] = 1;
        oneHotVectors.push(oneHotVector);
    }

    return oneHotVectors;
}

export function preprocessImages(images) {
    let result = []
    for (const image of images) {
        result.push(Array.from(image).map(v => v / 255.0));
    }
    return result;
}

export class Layer {
    weights;
    biases;
    forward() {
        throw Error('not implement')
    }
    backward() {
        throw Error('not implement')
    }
}

export class Dense extends Layer {
    inputs;
    outputs;
    gradBiases;
    gradWeights;
    gradInputs;
    active;
    activeFunc;
    derivativeFunc;
    constructor(input_size, output_size, active = 'sigmoid') {
        super();
        this.active = active
        switch (this.active) {
            case 'sigmoid':
                this.activeFunc = sigmoid;
                this.derivativeFunc = sigmoidDerivative;
                break;
            case 'relu':
                this.activeFunc = relu;
                this.derivativeFunc = reluDerivative;
                break;
            case 'leaky_relu':
                this.activeFunc = leakyRelu;
                this.derivativeFunc = leakyReluDerivative;
                break;
            case 'softmax':
                this.activeFunc = softmax;
                this.derivativeFunc = softmaxDerivative;
                break;
            default:
                break;
        }
        if (initHE.includes(this.active)) {
            this.weights = heInit(input_size, output_size);
        } else {
            this.weights = xavierInit(input_size, output_size);
        }
        // this.weights = Array.from({length: input_size}, () => Array.from({length: output_size}, () => getRandomArbitrary(-1,1)))
        this.biases = Array.from({ length: output_size }, () => getRandomArbitrary(-1, 1))
    }
    forward(inputs) {
        // save input for backward
        this.inputs = inputs;
        this.outputs = [];
        for (const [i, bias] of this.biases.entries()) {
            /* weights look like:
                [[w00,w01,w02],
                [w10,w11,w12]]*/
            // so we need do it like: w[0][0] * input[0] + w[1][0] * input[1] + ... + w[n][0] * input[n] (n = inputs.length - 1)
            let z = sumArr(this.inputs.map((input, j) => this.weights[j][i] * input)) + bias;
            this.outputs.push(z);
        }
        this.outputs = this.activeFunc(this.outputs);
        return this.outputs;
    }

    backward(grad, lr) {
        this.gradInputs = new Array(this.inputs.length).fill(0); // calculate gradient for previous layer
        this.gradWeights = this.inputs.map(() => new Array(this.outputs.length).fill(0)); // calculate gradient weights for this layer
        this.gradBiases = new Array(this.outputs.length).fill(0); // calculate gradient biases for this layer
        let derivativeOutputs = this.derivativeFunc(this.outputs);
        for (let i = 0; i < this.inputs.length; i++) {
            for (let j = 0; j < this.outputs.length; j++) {
                let derivativeValue = derivativeOutputs[j];
                this.gradInputs[i] += grad[j] * this.weights[i][j] * derivativeValue;
                this.gradWeights[i][j] = grad[j] * this.inputs[i] * derivativeValue;
                this.gradBiases[j] += grad[j] * derivativeValue;
                // update weights
                this.weights[i][j] -= lr * this.gradWeights[i][j];
                if (i === 0) {
                    this.biases[j] -= lr * this.gradBiases[j];
                }
            }
        }
        return this.gradInputs; // return for previous layer using to calculate gradient
    }
}

mappedClasses.set('Dense', Dense);

export class Model {
    loss;
    earlyStoppingCb;
    epochEndCb;
    monitors;
    outputs;
    layers = [];
    lossHistory = [];
    lossFunc;
    lossFuncGradient;
    constructor(loss, earlyStoppingCb, epochEndCb, monitors) {
        this.loss = loss;
        this.earlyStoppingCb = earlyStoppingCb;
        this.epochEndCb = epochEndCb;
        this.monitors = monitors;
        if (loss === 'categorical_crossentropy') {
            this.lossFunc = categoricalCrossEntropy;
            this.lossFuncGradient = categoricalCrossEntropyGradient;
        }
        if (loss === 'mse') {
            this.lossFunc = mse;
            this.lossFuncGradient = mseGradient;
        }
        if (loss === 'binary_crossentropy') {
            this.lossFunc = binaryCrossEntropy;
            this.lossFuncGradient = binaryCrossEntropyGradient;
        }
    }
    add(layer) {
        this.layers.push(layer);
    }
    forward(inputs) {
        let outputs = inputs;
        for (const layer of this.layers) {
            outputs = layer.forward(outputs)
        }
        return outputs;
    }

    backward(targets, lr) {
        let grad = this.lossFuncGradient(this.outputs, targets);
        // calculate gradient from last layer to first layer
        for (let i = this.layers.length - 1; i >= 0; i--) {
            grad = this.layers[i].backward(grad, lr)
        }
    }

    train(inputs, targets, epochs, learningRate) {
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;
            let accuracy = 0;
            // training each epoch
            const bar = new cliProgress.SingleBar({
                format: '{value}/{total} |' + colors.cyan('{bar}') + '| {percentage}% | Loss: {loss} | Accuracy: {accuracy}%',
                barCompleteChar: '\u2588',
                barIncompleteChar: '\u2591',
                hideCursor: true
            });
            bar.start(inputs.length, 1, { loss: 0, accuracy: 0 });
            for (let i = 0; i < targets.length; i++) {
                // propagation
                this.outputs = this.forward(inputs[i]);
                totalLoss += sumArr(this.lossFunc(this.outputs, targets[i]));
                this.backward(targets[i], learningRate);
                // count accuracy
                let pred = this.outputs.indexOf(Math.max(...this.outputs));
                let target = targets[i].indexOf(Math.max(...targets[i]));
                if (pred === target) {
                    accuracy++;
                }
                // show progress
                let count = i + 1;
                if (count % 100 === 0 && this.monitors && this.monitors.includes('progress')) {
                    bar.update(count, { loss: totalLoss / count, accuracy: (accuracy * 100.0 / count).toFixed(2) });
                }
            }
            bar.stop();
            // avg loss
            let loss = totalLoss / inputs.length;
            this.lossHistory.push(loss);
            if (this.monitors && this.monitors.includes('loss')) {
                console.log(`Epoch: ${epoch}, Loss: ${loss}, Accuracy: ${(accuracy * 100.0 / inputs.length).toFixed(2)}%`);
            }

            // callbacks
            typeof this.epochEndCb === 'function' && this.epochEndCb(this, epoch, loss);
            if (typeof this.earlyStoppingCb === 'function' && this.earlyStoppingCb(this, epoch, loss)) {
                console.log('Early stopping !');
                break;
            };

        }
    }

    summary() {
        console.log(`====================`)
        let totalParams = 0;
        for (const layer of this.layers) {
            let layerParams = layer.weights.length * layer.weights[0].length;
            totalParams += layerParams
            console.log(`${layer.constructor.name} (${layer.weights.length}, ${layer.weights[0].length}) Active: '${layer.active}' Params: ${layerParams}`)
        }
        console.log(`Total Params: ${totalParams}`)
        console.log(`====================`)
    }

    save(path) {
        const data = {
            classes: this.layers.map(l => l.constructor.name),
            layers: this.layers
        };
        writeFile(path, JSON.stringify(data));
        console.log(`Saved: ${path}`)
    }

    load(path) {
        let data = JSON.parse(readFile(path));
        const classes = data.classes;
        const layers = data.layers;
        const layersTmp = [];
        for (let i = 0; i < layers.length; i++) {
            const layerClass = mappedClasses.get(classes[i]);
            let layer = new layerClass(layers[i].input_size, layers[i].output_size, layers[i].active);
            layer.weights = layers[i].weights;
            layer.biases = layers[i].biases;
            layersTmp.push(layer);
        }
        this.layers = layersTmp;
        console.log(`Loaded: ${path}`);
        this.summary();
    }
}

function parseImages(buffer) {
    const header = buffer.slice(0, 16);
    const magic = header.readInt32BE(0);
    const numImages = header.readInt32BE(4);
    const rows = header.readInt32BE(8);
    const cols = header.readInt32BE(12);

    const data = [];
    let offset = 16;
    for (let i = 0; i < numImages; i++) {
        const image = new Uint8Array(buffer.slice(offset, offset + rows * cols));
        data.push(image);
        offset += rows * cols;
    }

    return data;
}

function parseLabels(buffer) {
    const header = buffer.slice(0, 8);
    const magic = header.readInt32BE(0);
    const numLabels = header.readInt32BE(4);

    const data = new Uint8Array(buffer.slice(8));
    return data;
}
export function loadMNIST() {
    // Load training data
    const trainImagesBuf = fs.readFileSync('./mnist-data/train-images.idx3-ubyte');
    const trainLabelsBuf = fs.readFileSync('./mnist-data/train-labels.idx1-ubyte');

    // Load test data
    const testImagesBuf = fs.readFileSync('./mnist-data/t10k-images.idx3-ubyte');
    const testLabelsBuf = fs.readFileSync('./mnist-data/t10k-labels.idx1-ubyte');


    const trainImages = parseImages(trainImagesBuf);
    const trainLabels = parseLabels(trainLabelsBuf);
    const testImages = parseImages(testImagesBuf);
    const testLabels = parseLabels(testLabelsBuf);
    return { trainImages, trainLabels, testImages, testLabels };
}

const { trainImages, trainLabels, testImages, testLabels } = loadMNIST();
// preprocess images
const xTrain = preprocessImages(trainImages);
const xTest = preprocessImages(testImages);

// preprocess labels (convert to one hot vector)
const yTrain = convertToOneHot(trainLabels, 10);
const yTest = convertToOneHot(testLabels, 10);

const epochEndCallback = (model, epoch, loss) => {
    if (epoch === 0 || loss < model.lossHistory[epoch - 1]) {
        model.save('./mnist_model.any_extension');
    }
}

const earlyStoppingCallback = (model, epoch, loss) => {
    return loss < 0.3;
}
let model = new Model('categorical_crossentropy', earlyStoppingCallback, epochEndCallback, ['loss', 'progress']);
model.add(new Dense(784, 10, 'leaky_relu'));
model.add(new Dense(10, 10, 'softmax'));
model.summary();

model.train(xTrain, yTrain, 10, 0.001);
let acc = 0;
for (let idx = 0; idx < yTest.length; idx++) {
    let output = model.forward(xTest[idx]);
    if (testLabels[idx] === output.indexOf(Math.max(...output))) {
        acc++;
    }
}

console.log(`Test Accuracy: ${acc} / ${testLabels.length} ~ ${(acc * 100.0 / testLabels.length).toFixed(2)}%`);