package org.example;

import java.io.Serializable;
import java.util.Random;

public class PerceptronLayer implements Serializable {
    private final int inputSize;
    private final int neuronCount;

    private final double[][] weights;
    private final double[] biases;

    private double[][] weightGradients;
    private double[] biasGradients;

    private final double[] outputs;
    private double[] lastInput;

    private final boolean outputLayer;

    public PerceptronLayer(int inputSize, int neuronCount, boolean outputLayer) {
        this.inputSize = inputSize;
        this.neuronCount = neuronCount;
        this.outputLayer = outputLayer;

        weights = new double[neuronCount][inputSize];
        biases = new double[neuronCount];
        outputs = new double[neuronCount];
        weightGradients = new double[neuronCount][inputSize];
        biasGradients = new double[neuronCount];

        initializeWeights();
    }

    private void initializeWeights() {
        Random random = new Random();
        double limit = Math.sqrt(6.0 / (inputSize + neuronCount));

        for (int n = 0; n < neuronCount; n++) {
            for (int i = 0; i < inputSize; i++) {
                weights[n][i] = random.nextDouble() * 2 * limit - limit;
            }
            biases[n] = 0.0;
        }
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double output) {
        return output * (1.0 - output);
    }

    public double[] feedForward(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Vector parameters mismatch");
        }
        lastInput = input.clone();
        for (int n = 0; n < neuronCount; n++) {
            double sum = biases[n];
            for (int i = 0; i < inputSize; i++) {
                sum += weights[n][i] * input[i];
            }
            outputs[n] = outputLayer ? sum : sigmoid(sum);
        }
        return outputs.clone();
    }

    public double[] backward(double[] outputGradient, double learningRate) {
        double[] inputGradient = new double[inputSize];
        double[] deltas = new double[neuronCount];

        for (int n = 0; n < neuronCount; n++) {
            double activationDerivative = outputLayer ? 1.0 : sigmoidDerivative(outputs[n]);
            deltas[n] = outputGradient[n] * activationDerivative;
        }

        for (int i = 0; i < inputSize; i++) {
            for (int n = 0; n < neuronCount; n++) {
                inputGradient[i] += weights[n][i] * deltas[n];
            }
        }

        for (int n = 0; n < neuronCount; n++) {
            for (int i = 0; i < inputSize; i++) {
                weights[n][i] -= learningRate * deltas[n] * lastInput[i];
            }
            biases[n] -= learningRate * deltas[n];
        }
        return inputGradient;
    }

    public void resetGradients() {
        for (int n = 0; n < neuronCount; n++) {
            biasGradients[n] = 0.0;
            for (int i = 0; i < inputSize; i++) {
                weightGradients[n][i] = 0.0;
            }
        }
    }

    public double[] accumulateGradients(double[] outputGradient) {
        double[] inputGradient = new double[inputSize];
        for (int n = 0; n < neuronCount; n++) {
            double derivative;
            if (outputLayer) {
                derivative = 1.0;
            } else {
                derivative = outputs[n] * (1.0 - outputs[n]);
            }
            double delta = outputGradient[n] * derivative;
            biasGradients[n] += delta;
            for (int i = 0; i < inputSize; i++) {
                inputGradient[i] += weights[n][i] * delta;
                weightGradients[n][i] += delta * lastInput[i];
            }
        }
        return inputGradient;
    }

    public void applyGradients(double step) {
        for (int n = 0; n < neuronCount; n++) {
            biases[n] -= step * biasGradients[n];
            for (int i = 0; i < inputSize; i++) {
                weights[n][i] -= step * weightGradients[n][i];
            }
        }
    }

    public double[] getParameters() {
        double[] result = new double[neuronCount * (inputSize + 1)];
        int index = 0;
        for (int n = 0; n < neuronCount; n++) {
            for (int i = 0; i < inputSize; i++) {
                result[index++] = weights[n][i];
            }
            result[index++] = biases[n];
        }
        return result;
    }

    public void setParameters(double[] parameters) {
        int index = 0;
        for (int n = 0; n < neuronCount; n++) {
            for (int i = 0; i < inputSize; i++) {
                weights[n][i] = parameters[index++];
            }
            biases[n] = parameters[index++];
        }
    }

    public double[] getGradients() {
        double[] result = new double[neuronCount * (inputSize + 1)];
        int index = 0;
        for (int n = 0; n < neuronCount; n++) {
            for (int i = 0; i < inputSize; i++) {
                result[index++] = weightGradients[n][i];
            }
            result[index++] = biasGradients[n];
        }
        return result;
    }

}
