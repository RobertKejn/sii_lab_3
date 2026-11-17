package org.example;

import javax.swing.*;
import java.awt.*;

class PlotPanel extends JPanel {

    private final double[][] expected;
    private final double[][] predicted;

    public PlotPanel(double[][] expected, double[][] predicted) {
        if (expected == null || predicted == null ||
                expected.length == 0 ||
                expected.length != predicted.length) {
            throw new IllegalArgumentException("Некорректные массивы данных");
        }

        this.expected = expected;
        this.predicted = predicted;

        setPreferredSize(new Dimension(900, 500));
        setBackground(Color.WHITE);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(
                RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON
        );

        int width = getWidth();
        int height = getHeight();

        int padding = 50;

        // Находим минимальное и максимальное значения
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;

        for (int i = 0; i < expected.length; i++) {
            min = Math.min(min, expected[i][0]);
            min = Math.min(min, predicted[i][0]);

            max = Math.max(max, expected[i][0]);
            max = Math.max(max, predicted[i][0]);
        }

        // Защита от одинаковых значений
        if (max == min) {
            max += 1;
            min -= 1;
        }

        // Рисуем оси
        g2.setColor(Color.BLACK);
        g2.drawLine(padding, padding, padding, height - padding);
        g2.drawLine(padding, height - padding,
                width - padding, height - padding);

        // Подписи осей
        g2.drawString("Y", 15, padding);
        g2.drawString("X", width - padding + 10, height - padding);

        // Рисуем графики
        drawLine(g2, expected, Color.BLUE, min, max, padding);
        drawLine(g2, predicted, Color.RED, min, max, padding);

        // Легенда
        g2.setColor(Color.BLUE);
        g2.drawLine(width - 180, 25, width - 150, 25);
        g2.drawString("Эталон", width - 145, 30);

        g2.setColor(Color.RED);
        g2.drawLine(width - 180, 45, width - 150, 45);
        g2.drawString("Прогноз", width - 145, 50);
    }

    private void drawLine(
            Graphics2D g2,
            double[][] data,
            Color color,
            double min,
            double max,
            int padding
    ) {
        int width = getWidth();
        int height = getHeight();

        int count = data.length;

        g2.setColor(color);
        g2.setStroke(new BasicStroke(2));

        for (int i = 0; i < count - 1; i++) {
            int x1 = padding + i * (width - 2 * padding) / (count - 1);
            int x2 = padding + (i + 1) * (width - 2 * padding) / (count - 1);

            int y1 = height - padding -
                    (int) ((data[i][0] - min) / (max - min)
                            * (height - 2 * padding));

            int y2 = height - padding -
                    (int) ((data[i + 1][0] - min) / (max - min)
                            * (height - 2 * padding));

            g2.drawLine(x1, y1, x2, y2);
        }
    }

    // Удобный запуск в отдельном окне
    public static void show(double[][] expected, double[][] predicted) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Сравнение прогноза");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.add(new PlotPanel(expected, predicted));
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
