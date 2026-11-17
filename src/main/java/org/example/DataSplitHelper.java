package org.example;

public class DataSplitHelper {
    public static double[][] splitDataForSlidingWindowNoPredict(double[] data, int windowSize) {
        int outputLength = data.length - windowSize + 1;
        double[][] outputData = new double[outputLength][windowSize];
        for (int i = 0; i < outputLength; i++) {
            System.arraycopy(data, i, outputData[i], 0, windowSize);
        }
        return outputData;
    }

    public static Object[] splitDataForSlidingWindowWithPredict(double[] data, int windowSize, int referenceSize, double test) {
        int outputLength = data.length - windowSize - referenceSize + 1;
        int outputLengthTest = (int) (outputLength * test);

        double[][] outputData = new double[outputLength - outputLengthTest][windowSize];
        double[][] outputRefData = new double[outputLength - outputLengthTest][referenceSize];

        double[][] outputDataTest = new double[outputLengthTest][windowSize];
        double[][] outputRefDataTest = new double[outputLengthTest][referenceSize];

        for (int i = 0; i < outputLength; i++) {
            if (i < outputLength - outputLengthTest) {
                System.arraycopy(data, i, outputData[i], 0, windowSize);
                System.arraycopy(data, i + windowSize, outputRefData[i], 0, referenceSize);
            } else {
                System.arraycopy(data, i, outputDataTest[i - (outputLength - outputLengthTest)], 0, windowSize);
                System.arraycopy(data, i + windowSize, outputRefDataTest[i - (outputLength - outputLengthTest)], 0, referenceSize);
            }
        }
        return new Object[]{outputData, outputRefData, outputDataTest, outputRefDataTest};
    }
}
