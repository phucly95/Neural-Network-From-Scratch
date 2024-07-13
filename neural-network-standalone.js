const initHE = ['relu', 'leaky_relu', 'softmax'];

// sum all items in array 2d
function sumArr(arr) {
    return arr.reduce((acc, curr) => acc + curr, 0);
}
// random in range [min,max]
function getRandomArbitrary(min, max) {
    return Math.random() * (max - min) + min;
}

function randn() {
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
function transpose(weights) {
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

function heInit(fanIn, fanOut) {
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

function xavierInit(fanIn, fanOut) {
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
    return arr.map(x => x > 0 ? 1 : 0);
}

export function leakyRelu(arr, alpha = 0.001) {
    return arr.map(x => x > 0 ? x : alpha * x);
}

export function leakyReluDerivative(arr, alpha = 0.001) {
    return arr.map(x => x > 0 ? 1 : alpha);
}

// softmax
function softmax(arr) {
    let maxX = Math.max(...arr);
    let expX = arr.map((v) => Math.exp(v - maxX));
    let sum = expX.reduce((acc, curr) => acc + curr, 0);
    return expX.map((v) => v / sum);
}
function softmaxDerivative(arr) {
    let s = softmax(arr);
    return arr.map((v, idx) => s[idx] * (1 - s[idx]));
}
// MSE loss
function mse(y_pred, y_true) {
    return y_pred.map((_, idx) => (y_pred[idx] - y_true[idx]) ** 2);
}
function mseGradient(y_pred, y_true) {
    return y_pred.map((_, idx) => (y_pred[idx] - y_true[idx]));
}
// Categorical Cross Entropy loss
function categoricalCrossEntropy(y_pred, y_true) {
    return y_true.map((_, i) => (-y_true[i] * Math.log(y_pred[i])));
}

function categoricalCrossEntropyGradient(y_pred, y_true) {
    return y_true.map((_, i) => (y_pred[i] - y_true[i]));
}

function binaryCrossEntropy(y_pred, y_true) {
    let loss = [];
    for (let i = 0; i < y_true.length; i++) {
        let currentLoss = y_true[i] === 1 ? -Math.log(y_pred[i]) : -Math.log(1 - y_pred[i]);
        loss.push(currentLoss);
    }
    return loss;
}
function binaryCrossEntropyGradient(y_pred, y_true) {
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

class Layer {
    weights;
    biases;
    forward() {
        throw Error('not implement')
    }
    backward() {
        throw Error('not implement')
    }
}

class Dense extends Layer {
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
        let transposed = transpose(this.weights);
        this.outputs = [];
        for (const [i, bias] of this.biases.entries()) {
            let z = sumArr(this.inputs.map((input, j) => transposed[i][j] * input)) + bias;
            this.outputs.push(z);
        }
        this.outputs = this.activeFunc(this.outputs);
        return this.outputs;
    }

    backward(grad, lr) {
        this.gradInputs = new Array(this.inputs.length).fill(0);
        this.gradWeights = this.inputs.map(() => new Array(this.outputs.length).fill(0));
        this.gradBiases = new Array(this.outputs.length).fill(0);
        let outputs = this.derivativeFunc(this.outputs);
        for (let i = 0; i < this.inputs.length; i++) {
            for (let j = 0; j < this.outputs.length; j++) {
                let derivativeValue = outputs[j];
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
        return this.gradInputs;
    }
}

class Model {
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
        switch (loss) {
            case 'categorical_crossentropy':
                this.lossFunc = categoricalCrossEntropy;
                this.lossFuncGradient = categoricalCrossEntropyGradient;
                break;
            case 'mse':
                this.lossFunc = mse;
                this.lossFuncGradient = mseGradient;
                break;
            case 'binary_crossentropy':
                this.lossFunc = binaryCrossEntropy;
                this.lossFuncGradient = binaryCrossEntropyGradient;
                break;
            default:
                break;
        }
    }
    add(layer) {
        this.layers.push(layer);
    }
    forward(inputs) {
        let outputs = [...inputs];
        for (const layer of this.layers) {
            outputs = layer.forward(outputs)
        }
        return outputs;
    }

    backward(targets, lr) {
        let grad = this.lossFuncGradient(this.outputs, targets);
        for (let i = this.layers.length - 1; i >= 0; i--) {
            grad = this.layers[i].backward(grad, lr)
        }
    }

    train(inputs, targets, epochs, learningRate) {
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;
            // training each epoch
            for (let i = 0; i < targets.length; i++) {
                // propagation
                this.outputs = this.forward(inputs[i]);
                totalLoss += sumArr(this.lossFunc(this.outputs, targets[i]));
                this.backward(targets[i], learningRate);
            }
            // avg loss
            let loss = totalLoss / inputs.length;
            this.lossHistory.push(loss);
            if (this.monitors && this.monitors.includes('loss')) {
                console.log(`Epoch: ${epoch}, Loss: ${loss}`);
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
}

const epochEndCallback = (model, epoch, loss) => {
    if (epoch === 0 || loss < model.lossHistory[epoch - 1]) {
        // model.save('./simple-model.any_extension');
    }
}

const earlyStoppingCallback = (model, epoch, loss) => {
    return loss < 0.0003;
}

let model = new Model('mse', earlyStoppingCallback, epochEndCallback, ['loss', 'progress']);
model.add(new Dense(2, 3, 'leaky_relu'));
model.add(new Dense(3, 1, 'sigmoid'));
model.summary()

let xTrain = [[0, 0], [0, 1], [1, 0], [1, 1]];
let yTrain = [[0], [1], [1], [0]];
model.train(xTrain, yTrain, 1000, 0.1);
for (let idx = 0; idx < yTrain.length; idx++) {
    let output = model.forward(xTrain[idx]);
    console.log(output)
}
