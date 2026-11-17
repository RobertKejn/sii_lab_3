package org.example;

import java.io.*;
import java.math.BigDecimal;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

public class Main {

    public static double[] generateSeries(int length) {
        double[] data = new double[length];

        for (int t = 0; t < length; t++) {
            data[t] = 10
                    + 5 * Math.sin(0.15 * t)
                    + 2 * Math.sin(0.04 * t)
                    + 0.02 * t;
        }

        return data;
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        double[][] data = CsvDataHelper.readCsv("aapl_raw_data.csv", new HashSet<>(List.of(0)));

        int targetColumn = 4;
        int window = 20;
        int clusters = 200;
        int hidden = 120;
        int outputs = 1;
        double testSize = 0.2;
        int epoch = 50;
        BigDecimal threshold = new BigDecimal("0.000001");

        Object[] split = CsvDataHelper.splitData(data, window, targetColumn, testSize);

        double[][] trainInput = (double[][]) split[0];
        double[][] trainOutput = (double[][]) split[1];

        double[][] testInput = (double[][]) split[2];
        double[][] testOutput = (double[][]) split[3];


        String networkPath = String.format("bad_network_w%d_c%d_e%d_h%d_t%d", window, clusters, epoch, hidden, threshold.scale()+1); //"bad_network_w20_c200_e50_t6";
        NeuralNetwork network;
        boolean saveNetwork = true;
        if (saveNetwork) {
            FuzzyLayer fuzzyLayer = new FuzzyLayer(clusters, trainInput[0].length);
            fuzzyLayer.loadData(trainInput);
            fuzzyLayer.fit(threshold.doubleValue());

            network = new NeuralNetwork(
                    new int[]{clusters, hidden, outputs},
                    fuzzyLayer
            );

            network.fit(
                    trainInput,
                    trainOutput,
                    epoch,
                    1
            );
            saveNetwork(network, networkPath);
        } else network = readNetwork(networkPath);


        double[][] results = new double[testInput.length][1];
        for (int i = 0; i < testInput.length; i++) {
            double[] res = network.feedForward(testInput[i]);
            double pred = res[0];
            double reference = testOutput[i][0];
            //results[i][0] = reference;
            results[i][0] = pred;
            System.out.println(Arrays.toString(testInput[i]) + "  -> " + pred + " |  ref: " + reference + " diff: " + (reference - pred));
        }
        PlotPanel.show(testOutput, results);

    }
    public static void main1(String[] args) throws IOException, ClassNotFoundException {
        int window = 2;
        int clusters = 60;
        int hidden = 100;
        int outputs = 1;
        int samples = 2000;
        double testSize = 1;
        int epoch = 30;
        BigDecimal threshold = new BigDecimal("0.000001");

        double[] dataRaw = generateSeries(samples);
        Object[] split = DataSplitHelper.splitDataForSlidingWindowWithPredict(dataRaw, window, 1, testSize);

        double[][] trainInput = (double[][]) split[0];
        double[][] trainOutput = (double[][]) split[1];

        double[][] testInput = (double[][]) split[2];
        double[][] testOutput = (double[][]) split[3];

        String networkPath = String.format("sin_w%d_c%d_e%d_h%d_t%d", window, clusters, epoch, hidden, threshold.scale()+1);
        NeuralNetwork network;
        boolean saveNetwork = false;
        if (saveNetwork) {
            FuzzyLayer fuzzyLayer = new FuzzyLayer(clusters, trainInput[0].length);
            fuzzyLayer.loadData(trainInput);
            fuzzyLayer.fit(threshold.doubleValue());

            network = new NeuralNetwork(
                    new int[]{clusters, hidden, outputs},
                    fuzzyLayer
            );

            network.fit(
                    trainInput,
                    trainOutput,
                    epoch,
                    1
            );
            saveNetwork(network, networkPath);
        } else network = readNetwork(networkPath);

        double[][] results = new double[testInput.length][1];
        for (int i = 0; i < testInput.length; i++) {
            double[] res = network.feedForward(testInput[i]);
            double pred = res[0];
            double reference = testOutput[i][0];
            //results[i][0] = reference;
            results[i][0] = pred;
            System.out.println(Arrays.toString(testInput[i]) + "  -> " + pred + " |  ref: " + reference + " diff: " + (reference - pred));
        }
        PlotPanel.show(testOutput, results);
    }

    static void saveNetwork(NeuralNetwork network, String path) throws IOException {
        ObjectOutputStream ous = new ObjectOutputStream(new FileOutputStream(path));
        ous.writeObject(network);
        ous.flush();
        ous.close();
    }

    static NeuralNetwork readNetwork(String path) throws IOException, ClassNotFoundException {
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path));
        NeuralNetwork neuralNetwork = (NeuralNetwork) ois.readObject();
        ois.close();
        return neuralNetwork;
    }
}


