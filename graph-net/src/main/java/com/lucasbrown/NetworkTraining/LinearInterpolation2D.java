package com.lucasbrown.NetworkTraining;

import java.util.function.ToDoubleBiFunction;
import java.util.stream.IntStream;

/**
 * Generates and linearly interpolates points in 2D using the provided
 * generating function.
 * The evaluation points are generated as the outer product of the given range
 * objects.
 * The range objects must be given in the same order as expected by the
 * interpolation call.
 * i.e, ranges = [X, Y] then interpolate must also be called as
 * interpolate(x, y).
 * If interpolate is called with a point outside of the evaluation range, the
 * generating function will be called on the provided data point.
 * 
 */
public class LinearInterpolation2D implements IInterpolation {

    private final Range x_range;
    private final Range y_range;
    private final ToDoubleBiFunction<Double, Double> generatingFunction;
    private final double[][] samplePoints;

    public LinearInterpolation2D(Range x_range, Range y_range,
            final ToDoubleBiFunction<Double, Double> generatingFunction, boolean parallelize) {
        this.x_range = x_range;
        this.y_range = y_range;
        this.generatingFunction = generatingFunction;
        samplePoints = generateSamples(parallelize);
    }

    public LinearInterpolation2D(Range x_range, Range y_range,
            final ToDoubleBiFunction<Double, Double> generatingFunction) {
        this(x_range, y_range, generatingFunction, false);
    }

    private double[][] generateSamples(boolean parallelize) {
        final int x_len = x_range.getNumberOfPoints();
        final int y_len = y_range.getNumberOfPoints();

        IntStream stream = IntStream.range(0, x_len);
        if (parallelize) {
            stream = stream.parallel();
        }

        return stream.mapToObj(i -> {
            final double xi = x_range.getValue(i);

            return IntStream.range(0, y_len).mapToDouble(j -> {
                double yj = y_range.getValue(j);
                return generatingFunction.applyAsDouble(xi, yj);
            }).toArray();

        }).toArray(double[][]::new);
    }

    /**
     * Linearly interpolate the data at the point x.
     * Extrapolation of points will use the generating function and will not expand
     * the evaluation range
     * 
     * @param x A point to interpolate
     * @return The interpolated value
     */
    @Override
    public double interpolate(double... point) {
        double x = point[0];
        double y = point[1];

        if (!x_range.isWithinRange(x) || !y_range.isWithinRange(y)) {
            return generatingFunction.applyAsDouble(x, y);
        }

        int i = x_range.getNearestIndex(x);
        int j = y_range.getNearestIndex(y);

        double wx = x_range.getIndexResidualWeight(x);
        double wy = y_range.getIndexResidualWeight(y);

        // account for the edge case where x or y are exactly 0
        if (i == -1) {
            i++;
            wx += 1;
        }

        if (j == -1) {
            j++;
            wy += 1;
        }

        return wx * wy * samplePoints[i][j]
                + (1 - wx) * wy * samplePoints[i + 1][j]
                + wx * (1 - wy) * samplePoints[i][j + 1]
                + (1 - wx) * (1 - wy) * samplePoints[i + 1][j + 1];
    }

    @Override
    public double[] interpolateDerivative(double... point) {
        double x = point[0];
        double y = point[1];

        // this is probably bad but eh
        if (!x_range.isWithinRange(x) || !y_range.isWithinRange(y)) {
            final double h = 0.1;
            return new double[] {
                    (generatingFunction.applyAsDouble(x + h / 2d, y) - generatingFunction.applyAsDouble(x - h / 2d, y))
                            / h,
                    (generatingFunction.applyAsDouble(x, y + h / 2d) - generatingFunction.applyAsDouble(x, y - h / 2d))
                            / h };
        }

        int i = x_range.getNearestIndex(x);
        int j = y_range.getNearestIndex(y);

        double wx = 1 - x_range.getIndexResidualWeight(x);
        double wy = 1 - y_range.getIndexResidualWeight(y);

        double dx = x_range.getValue(i + 1) - x_range.getValue(i);
        double dy = y_range.getValue(j + 1) - y_range.getValue(j);

        return new double[] {
                (wy * (samplePoints[i + 1][j] - samplePoints[i][j])
                        + (1 - wy) * (samplePoints[i + 1][j + 1] - samplePoints[i][j + 1])) / dx,
                (wx * (samplePoints[i][j + 1] - samplePoints[i][j])
                        + (1 - wx) * (samplePoints[i + 1][j + 1] - samplePoints[i + 1][j])) / dy };
    }
}
