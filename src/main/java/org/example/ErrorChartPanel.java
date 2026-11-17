package org.example;

import javax.swing.*;
import java.awt.*;

public class ErrorChartPanel extends JPanel {

    private double[][] data;

    private final int left = 75;
    private final int right = 25;
    private final int top = 25;
    private final int bottom = 55;

    public ErrorChartPanel(double[][] data) {
        this.data = data;
        setBackground(Color.WHITE);
        setPreferredSize(new Dimension(700, 400));
    }

    public void setData(double[][] data) {
        this.data = data;
        repaint();
    }

    public static void show(double[][] errors){
        JFrame frame = new JFrame("Ошибка обучения");

        ErrorChartPanel chart = new ErrorChartPanel(errors);

        frame.add(chart);
        frame.setSize(800, 500);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        if (data == null || data.length == 0) {
            return;
        }

        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(
                RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON
        );

        int width = getWidth();
        int height = getHeight();

        int plotWidth = width - left - right;
        int plotHeight = height - top - bottom;

        if (plotWidth <= 0 || plotHeight <= 0) {
            g2.dispose();
            return;
        }

        double minX = data[0][0];
        double maxX = data[0][0];
        double maxY = 0;

        for (double[] point : data) {
            minX = Math.min(minX, point[0]);
            maxX = Math.max(maxX, point[0]);
            maxY = Math.max(maxY, point[1]);
        }

        // Защита от нулевого диапазона
        if (minX == maxX) maxX = minX + 1;
        if (maxY <= 0) maxY = 1;

        // Небольшой запас сверху
        maxY *= 1.1;

        // Подбираем удобный шаг градуировки
        double yStep = niceStep(maxY / 5.0);
        double xStep = niceStep((maxX - minX) / 6.0);

        maxY = Math.ceil(maxY / yStep) * yStep;

        // Преобразование координат данных в координаты экрана
        final double finalMinX = minX;
        final double finalMaxX = maxX;
        final double finalMaxY = maxY;

        java.util.function.DoubleFunction<Integer> mapX =
                x -> left + (int) ((x - finalMinX)
                        / (finalMaxX - finalMinX) * plotWidth);

        java.util.function.DoubleFunction<Integer> mapY =
                y -> top + plotHeight -
                        (int) (y / finalMaxY * plotHeight);

        // Сетка и подписи Y
        g2.setFont(new Font("Arial", Font.PLAIN, 11));

        for (double y = 0; y <= maxY; y += yStep) {
            int py = mapY.apply(y);

            g2.setColor(new Color(225, 225, 225));
            g2.drawLine(left, py, width - right, py);

            g2.setColor(Color.DARK_GRAY);
            g2.drawString(
                    String.format("%.4f", y),
                    5, py + 4
            );
        }

        // Сетка и подписи X
        for (double x = Math.ceil(minX / xStep) * xStep;
             x <= maxX; x += xStep) {

            int px = mapX.apply(x);

            g2.setColor(new Color(235, 235, 235));
            g2.drawLine(px, top, px, height - bottom);

            g2.setColor(Color.DARK_GRAY);
            g2.drawString(
                    String.format("%.0f", x),
                    px - 10, height - bottom + 18
            );
        }

        // Оси
        g2.setColor(Color.BLACK);
        g2.drawLine(left, top, left, height - bottom);
        g2.drawLine(left, height - bottom, width - right, height - bottom);

        // График
        g2.setColor(new Color(40, 110, 210));
        g2.setStroke(new BasicStroke(2f));

        for (int i = 1; i < data.length; i++) {
            int x1 = mapX.apply(data[i - 1][0]);
            int y1 = mapY.apply(data[i - 1][1]);

            int x2 = mapX.apply(data[i][0]);
            int y2 = mapY.apply(data[i][1]);

            g2.drawLine(x1, y1, x2, y2);
        }

        // Точки
        g2.setColor(new Color(20, 80, 180));

        for (double[] point : data) {
            int x = mapX.apply(point[0]);
            int y = mapY.apply(point[1]);
            g2.fillOval(x - 3, y - 3, 6, 6);
        }

        // Подписи осей
        g2.setColor(Color.BLACK);
        g2.setFont(new Font("Arial", Font.BOLD, 13));

        g2.drawString(
                "Эпоха",
                left + plotWidth / 2 - 20,
                height - 10
        );

        g2.rotate(-Math.PI / 2);
        g2.drawString(
                "MSE",
                -(top + plotHeight / 2 + 15),
                18
        );

        g2.dispose();
    }

    // Подбор удобного шага градуировки
    private double niceStep(double value) {
        if (value <= 0) return 1;

        double power = Math.pow(
                10, Math.floor(Math.log10(value))
        );

        double fraction = value / power;

        if (fraction <= 1) return power;
        if (fraction <= 2) return 2 * power;
        if (fraction <= 5) return 5 * power;

        return 10 * power;
    }
}

