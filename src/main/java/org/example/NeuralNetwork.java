package org.example;

import java.io.Serializable;

public class NeuralNetwork implements Serializable {
    private final FuzzyLayer fuzzyLayer;
    private final PerceptronLayer[] layers;

    private double[][] learningLog;

    private final int[] layerSizes;

    public NeuralNetwork(int[] layerSizes, FuzzyLayer fuzzyLayer) {
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("Need 2 layers");
        }

        this.layerSizes = layerSizes.clone();
        this.fuzzyLayer = fuzzyLayer;

        if (layerSizes[0] != fuzzyLayer.getClusterCount()) {
            throw new IllegalArgumentException("Clusters dont match layer count");
        }

        layers = new PerceptronLayer[layerSizes.length - 1];

        for (int i = 1; i < layerSizes.length; i++) {
            int inputSize = layerSizes[i - 1];
            int neuronCount = layerSizes[i];

            boolean isOutput = (i == layerSizes.length - 1);

            layers[i - 1] = new PerceptronLayer(inputSize, neuronCount, isOutput);
        }
    }

    public double[] feedForward(double[] input) {
        double[] result = fuzzyLayer.feedForward(input);
        for (PerceptronLayer layer : layers) {
            result = layer.feedForward(result);
        }
        return result;
    }

    public void fit(double[][] inputs, double[][] expected, int epochs, int logFrequency, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0.0;
            for (int i = 0; i < inputs.length; i++) {
                totalError += trainSample(inputs[i], expected[i], learningRate);
            }
            double averageError = totalError / inputs.length;
            if (epoch % logFrequency == 0) {
                System.out.println("Epoch: " + epoch + ", Error: " + averageError);
            }
        }
    }
    public double trainSample(double[] input, double[] expected, double learningRate) {
        double[] prediction = feedForward(input);
        if (expected.length != prediction.length) {
            throw new IllegalArgumentException("Train output mismatch length");
        }

        double error = 0.0;
        double[] gradient = new double[prediction.length];

        for (int i = 0; i < prediction.length; i++) {
            double diff = prediction[i] - expected[i];

            error += diff * diff;
            gradient[i] = 2.0 * diff / prediction.length;
        }

        error /= prediction.length;

        for (int i = layers.length - 1; i >= 0; i--) {
            gradient = layers[i].backward(gradient, learningRate);
        }
        return error;
    }

    //

    private double[] getParameters() {
        int size = 0;
        for (PerceptronLayer layer : layers) {
            size += layer.getParameters().length;
        }
        double[] result = new double[size];
        int index = 0;
        for (PerceptronLayer layer : layers) {
            double[] p = layer.getParameters();
            System.arraycopy(p, 0, result, index, p.length);
            index += p.length;
        }
        return result;
    }

    private void setParameters(double[] parameters) {
        int index = 0;

        for (PerceptronLayer layer : layers) {
            int size = layer.getParameters().length;

            double[] part = new double[size];
            System.arraycopy(parameters, index, part, 0, size);

            layer.setParameters(part);
            index += size;
        }
    }

    private double[] getGradients() {
        int size = 0;
        for (PerceptronLayer layer : layers) {
            size += layer.getGradients().length;
        }
        double[] result = new double[size];
        int index = 0;
        for (PerceptronLayer layer : layers) {
            double[] g = layer.getGradients();
            System.arraycopy(g, 0, result, index, g.length);
            index += g.length;
        }

        return result;
    }

    private double calculateError(double[][] input, double[][] expected) {
        double error = 0.0;
        for (int i = 0; i < input.length; i++) {
            double[] predicted = feedForward(input[i]);
            for (int j = 0; j < predicted.length; j++) {
                double diff = predicted[j] - expected[i][j];
                error += diff * diff;
            }
        }
        return error / (input.length * expected[0].length);
    }

    private void calculateGradients(double[][] input, double[][] expected) {
        for (PerceptronLayer layer : layers) {
            layer.resetGradients();
        }

        for (int i = 0; i < input.length; i++) {
            double[] predicted = feedForward(input[i]);
            double[] gradient = new double[predicted.length];
            for (int j = 0; j < predicted.length; j++) {
                gradient[j] = 2.0 * (predicted[j] - expected[i][j]) / (input.length * predicted.length);
            }

            for (int j = layers.length - 1; j >= 0; j--) {
                gradient = layers[j].accumulateGradients(gradient);
            }
        }
    }

    private double calculateErrorAtStep(double alpha, double[] baseParameters, double[] gradients, double[][] input, double[][] expected) {
        double[] trial = new double[baseParameters.length];
        for (int i = 0; i < trial.length; i++) {
            trial[i] = baseParameters[i] - alpha * gradients[i];
        }

        setParameters(trial);

        double error = calculateError(input, expected);

        setParameters(baseParameters);

        return error;
    }

    private double findOptimalStep(double[] baseParameters, double[] gradients, double[][] input, double[][] expected) {
        double left = 0.0;
        double right = 1.0;

        double baseError = calculateErrorAtStep(0.0, baseParameters, gradients, input, expected);
        double previousError = calculateErrorAtStep(right, baseParameters, gradients, input, expected);

        while (previousError < baseError && right < 1_000_000) {
            double nextRight = right * 2.0;
            double nextError = calculateErrorAtStep(nextRight, baseParameters, gradients, input, expected);
            if (nextError >= previousError) {
                right = nextRight;
                break;
            }
            right = nextRight;
            previousError = nextError;
        }

        double phi = (Math.sqrt(5.0) - 1.0) / 2.0;

        double x1 = right - phi * (right - left);
        double x2 = left + phi * (right - left);

        double f1 = calculateErrorAtStep(x1, baseParameters, gradients, input, expected);
        double f2 = calculateErrorAtStep(x2, baseParameters, gradients, input, expected);

        for (int i = 0; i < 50; i++) {
            if (f1 > f2) {
                left = x1;
                x1 = x2;
                f1 = f2;
                x2 = left + phi * (right - left);
                f2 = calculateErrorAtStep(x2, baseParameters, gradients, input, expected);
            } else {
                right = x2;
                x2 = x1;
                f2 = f1;
                x1 = right - phi * (right - left);
                f1 = calculateErrorAtStep(x1, baseParameters, gradients, input, expected);
            }
        }
        return (left + right) / 2.0;
    }

    public void fit(double[][] input, double[][] expected, int epochs, int logFrequency) {
        learningLog = new double[epochs][2];
        for (int epoch = 0; epoch < epochs; epoch++) {
            calculateGradients(input, expected);

            double[] parameters = getParameters();
            double[] gradients = getGradients();

            double normSquared = 0.0;
            for (double g : gradients) {
                normSquared += g * g;
            }
            if (normSquared < 1e-16) {
                System.out.println("Gradient is close to zero");
                break;
            }

            double step = findOptimalStep(parameters, gradients, input, expected);
            double[] newParameters = new double[parameters.length];
            for (int i = 0; i < parameters.length; i++) {
                newParameters[i] = parameters[i] - step * gradients[i];
            }
            setParameters(newParameters);

            double error = calculateError(input, expected);
            learningLog[epoch][0] = epoch;
            learningLog[epoch][1] = error;
            if (epoch % logFrequency == 0) {
                System.out.println("Epoch: " + epoch + ", Error MSE: " + error + ", Step: " + step);
            }
        }
    }

    public double[][] getLearningLog(){
        return learningLog;
    }
}
