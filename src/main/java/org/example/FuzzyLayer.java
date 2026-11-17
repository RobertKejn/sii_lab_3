package org.example;

import java.io.Serializable;
import java.util.Random;

public class FuzzyLayer implements Serializable {
    final int CLUSTERS;
    final int PARAMETERS_COUNT;

    int DATA_LENGTH;
    double[][] data;
    double[][] fuzzy;

    double[][] centroids;
    double[][] widths;

    public FuzzyLayer() {
        CLUSTERS = 2;
        PARAMETERS_COUNT = 2;
    }

    public FuzzyLayer(int clusters, int parameters) {
        if (clusters < 1 || parameters < 1) throw new Error("Non-positive hyper parameters");
        CLUSTERS = clusters;
        PARAMETERS_COUNT = parameters;
    }

    public void loadData(double[][] data) {
        DATA_LENGTH = data.length;
        this.data = new double[DATA_LENGTH][PARAMETERS_COUNT];
        fuzzy = new double[DATA_LENGTH][CLUSTERS];
        Random r = new Random();
        for (int i = 0; i < DATA_LENGTH; i++) {
            System.arraycopy(data[i], 0, this.data[i], 0, PARAMETERS_COUNT);
            double sum = 0;
            for (int j = 0; j < CLUSTERS; j++) {
                fuzzy[i][j] = r.nextDouble();
                sum += fuzzy[i][j];
            }
            for (int j = 0; j < CLUSTERS; j++) {
                fuzzy[i][j] /= sum;
            }
        }
    }

    public void fit(double threshold) {
        double m = 2;
        boolean flag;
        do {
            double[][] fuzzyNew = fuzzyIteration(fuzzy, m);
            flag = validateDifference(fuzzy, fuzzyNew, threshold);
            fuzzy = fuzzyNew;
        }
        while (flag);
        calculateCentroids(fuzzy, m);
        calculateWidths(fuzzy, m);
    }

    private boolean validateDifference(double[][] fuzzy, double[][] fuzzyNew, double threshold) {
        double diff = 0;
        for (int c = 0; c < CLUSTERS; c++) {
            for (int d = 0; d < DATA_LENGTH; d++) {
                diff += Math.abs(fuzzy[d][c] - fuzzyNew[d][c]);
            }
        }
        double avgDiff = diff / (CLUSTERS * DATA_LENGTH);
        System.out.println("avgDiff: " + avgDiff);
        return avgDiff > threshold;
    }


    private void calculateCentroids(double[][] membership, double m) {
        centroids = new double[CLUSTERS][PARAMETERS_COUNT];

        for (int c = 0; c < CLUSTERS; c++) {
            double denominator = 0;

            for (int d = 0; d < DATA_LENGTH; d++) {
                double u = Math.pow(membership[d][c], m);
                denominator += u;

                for (int p = 0; p < PARAMETERS_COUNT; p++) {
                    centroids[c][p] += u * data[d][p];
                }
            }

            if (denominator > 0) {
                for (int p = 0; p < PARAMETERS_COUNT; p++) {
                    centroids[c][p] /= denominator;
                }
            }
        }
    }

    private void calculateWidths(double[][] membership, double m) {
        widths = new double[CLUSTERS][PARAMETERS_COUNT];

        for (int c = 0; c < CLUSTERS; c++) {
            for (int p = 0; p < PARAMETERS_COUNT; p++) {
                double sumR = 0;
                double delR = 0;

                for (int d = 0; d < DATA_LENGTH; d++) {
                    double u = Math.pow(membership[d][c], m);
                    double diff = data[d][p] - centroids[c][p];

                    sumR += u * diff * diff;
                    delR += u;
                }

                widths[c][p] = Math.max(sumR / delR, 1e-9);
            }
        }
    }

    private double[][] fuzzyIteration(double[][] fuzzy, double m) {
        double[][] fuzzyPow = new double[DATA_LENGTH][CLUSTERS];
        double[] fuzzyDel = new double[CLUSTERS];
        for (int c = 0; c < CLUSTERS; c++) {
            for (int d = 0; d < DATA_LENGTH; d++) {
                fuzzyPow[d][c] = Math.pow(fuzzy[d][c], m);
                fuzzyDel[c] += fuzzyPow[d][c];
            }
        }

        centroids = new double[CLUSTERS][PARAMETERS_COUNT];
        for (int c = 0; c < CLUSTERS; c++) {
            for (int p = 0; p < PARAMETERS_COUNT; p++) {
                for (int d = 0; d < DATA_LENGTH; d++) {
                    centroids[c][p] += fuzzyPow[d][c] * data[d][p];
                }
                centroids[c][p] /= fuzzyDel[c];
            }
        }

        double[][] distances = new double[DATA_LENGTH][CLUSTERS];
        for (int d = 0; d < DATA_LENGTH; d++) {
            for (int c = 0; c < CLUSTERS; c++) {
                double distDtoC = 0;
                for (int p = 0; p < PARAMETERS_COUNT; p++) {
                    distDtoC += Math.pow(data[d][p] - centroids[c][p], 2);
                }
                distances[d][c] = Math.sqrt(distDtoC);
            }
        }

        double[][] fuzzyNew = new double[DATA_LENGTH][CLUSTERS];
        for (int d = 0; d < DATA_LENGTH; d++) {
            for (int c = 0; c < CLUSTERS; c++) {
                double sumDCallDist = 0;
                for (int cdist = 0; cdist < CLUSTERS; cdist++) {
                    sumDCallDist += (distances[d][c] * distances[d][c]) / (distances[d][cdist] * distances[d][cdist]);
                }
                double sumToPow = Math.pow(sumDCallDist, 1 / (double) (m - 1));
                fuzzyNew[d][c] = 1 / sumToPow;
            }
        }

        return fuzzyNew;
    }

    public double[] feedForward(double[] input) {
        if(input.length != PARAMETERS_COUNT) throw new Error("Parameters mismatch");
        double[] output = new double[CLUSTERS];
        double sum = 0.0;

        for (int c = 0; c < CLUSTERS; c++) {
            double exponent = 0.0;
            for (int p = 0; p < PARAMETERS_COUNT; p++) {
                double diff = input[p] - centroids[c][p];
                exponent += (diff * diff) / (2.0 * widths[c][p]);
            }
            output[c] = Math.exp(-exponent);
            sum += output[c];
        }

        if (sum > 0.0) {
            for (int c = 0; c < CLUSTERS; c++) {
                output[c] /= sum;
            }
        }
        return output;
    }

    public int getClusterCount(){
        return CLUSTERS;
    }

}
