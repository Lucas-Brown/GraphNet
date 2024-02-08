package com.lucasbrown.NetworkTraining;

import java.util.function.ToDoubleBiFunction;
import java.util.stream.IntStream;

public class LinearInterpolation2D implements Interpolation{

    private final Range x_range;
    private final Range y_range;
    private final ToDoubleBiFunction<Double, Double> generatingFunction;
    private final double[][] samplePoints;

    public LinearInterpolation2D(Range x_range, Range y_range, final ToDoubleBiFunction<Double, Double> generatingFunction)
    {
        this.x_range = x_range;
        this.y_range = y_range;
        this.generatingFunction = generatingFunction;
        samplePoints = generateSamples();
    }

    private double[][] generateSamples()
    {
        final int x_len = x_range.getNumberOfPoints();
        final int y_len = y_range.getNumberOfPoints();

        return IntStream.range(0, x_len).mapToObj(i -> 
        {
            final double xi = x_range.getValue(i);
            
            return IntStream.range(0, y_len).mapToDouble(j -> 
            {
                double yj = y_range.getValue(j);
                return generatingFunction.applyAsDouble(xi, yj);
            }).toArray();
            
        }).toArray(double[][]::new);
    }

    @Override
    public double interpolate(double... point) 
    {
        double x = point[0];
        double y = point[1];

        if(!x_range.isWithinRange(x) || !y_range.isWithinRange(y))
        {
            return generatingFunction.applyAsDouble(x, y);
        }

        int i = x_range.getNearestIndex(x);
        int j = y_range.getNearestIndex(y);

        double wx = x_range.getIndexResidualWeight(x);
        double wy = y_range.getIndexResidualWeight(y);

        // account for the edge case where x or y are exactly 0
        if(i == -1)
        {
            i++;
            wx += 1;
        }

        if(j == -1)
        {
            j++;
            wy += 1;
        }

        return wx*wy*samplePoints[i][j]
            + (1-wx)*wy*samplePoints[i+1][j]
            + wx*(1-wy)*samplePoints[i][j+1]
            + (1-wx)*(1-wy)*samplePoints[i+1][j+1];
    }

    @Override
    public double[] interpolateDerivative(double... point) 
    {
        double x = point[0];
        double y = point[1];

        // this is probably bad but eh
        if(!x_range.isWithinRange(x) || !y_range.isWithinRange(y))
        {
            final double h = 0.1;
            return new double[]{(generatingFunction.applyAsDouble(x+h/2d, y) - generatingFunction.applyAsDouble(x-h/2d, y)) / h, 
                (generatingFunction.applyAsDouble(x, y+h/2d) - generatingFunction.applyAsDouble(x, y-h/2d)) / h};
        }

        int i = x_range.getNearestIndex(x);
        int j = y_range.getNearestIndex(y);

        double wx = 1 - x_range.getIndexResidualWeight(x);
        double wy = 1 - y_range.getIndexResidualWeight(y);

        double dx = x_range.getValue(i+1) - x_range.getValue(i);
        double dy = y_range.getValue(j+1) - y_range.getValue(j);

        return new double[]{(wy * (samplePoints[i+1][j] - samplePoints[i][j]) + (1-wy) * (samplePoints[i+1][j+1] - samplePoints[i][j+1]))/dx,
            (wx * (samplePoints[i][j+1] - samplePoints[i][j]) + (1-wx) * (samplePoints[i+1][j+1] - samplePoints[i+1][j]))/dy};
    }
}
