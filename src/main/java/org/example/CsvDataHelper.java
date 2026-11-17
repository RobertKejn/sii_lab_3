package org.example;

import java.io.*;
import java.util.*;

public class CsvDataHelper {

    public static double[][] readCsv(String path, Set<Integer> skip) throws IOException {
        List<double[]> rows = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            String line;
            reader.readLine();
            while ((line = reader.readLine()) != null) {
                while(line.contains(",,")) line = line.replace(",,", ",0,");
                String[] values = line.split(",");
                double[] row = new double[values.length];

                for (int i = 0; i < values.length; i++) {
                    if (!skip.contains(i)) row[i] = Double.parseDouble(values[i].trim());
                }
                rows.add(row);
            }
        }

        return rows.toArray(new double[0][]);
    }

    public static Object[] splitData(double[][] data, int windowSize, int targetColumn, double testRatio) {

        int count = data.length - windowSize;

        int testCount = (int) (count * testRatio);
        int trainCount = count - testCount;

        int parameters = data[0].length;

        double[][] trainInput = new double[trainCount][windowSize * parameters];
        double[][] trainOutput = new double[trainCount][1];

        double[][] testInput = new double[testCount][windowSize * parameters];
        double[][] testOutput = new double[testCount][1];

        for (int i = 0; i < count; i++) {
            double[] input = new double[windowSize * parameters];

            for (int j = 0; j < windowSize; j++) {
                System.arraycopy(data[i + j], 0, input, j * parameters, parameters);
            }

            double target = data[i + windowSize][targetColumn];

            if (i < trainCount) {
                trainInput[i] = input;
                trainOutput[i][0] = target;
            } else {
                int index = i - trainCount;
                testInput[index] = input;
                testOutput[index][0] = target;
            }
        }

        return new Object[]{trainInput, trainOutput, testInput, testOutput};
    }
}
