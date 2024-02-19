package com.lucasbrown.NetworkTraining;

import java.util.Arrays;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Generates and linearly interpolates points using the provided
 * generating function.
 * The evaluation points are generated as the outer product of the given range
 * objects.
 * The range objects must be given in the same order as expected by the
 * interpolation call.
 * i.e, ranges = [X, Y, Z] then interpolate must also be called as
 * interpolate(x, y, z).
 * If interpolate is called with a point outside of the evaluation range, the
 * generating function will be called on the provided data point.
 */
public class LinearInterpolation implements IInterpolation {

    // The ordered ranges for evaluation
    private final Range[] ranges;

    // The generating function to evaluate
    private final ToDoubleFunction<double[]> generatingFunction;

    // The generated points to interpolate
    private final NDarray samplePoints;

    public LinearInterpolation(Range[] ranges, final ToDoubleFunction<double[]> generatingFunction) {
        this.ranges = ranges;
        this.generatingFunction = generatingFunction;
        samplePoints = generateSamples();
    }

    private NDarray generateSamples() {
        Stream<int[]> indexOuterProduct = Stream.of(new int[0]);
        int[] sizes = new int[ranges.length];

        for (int i = 0; i < ranges.length; i++) {
            final int j = i;
            sizes[j] = ranges[j].getNumberOfPoints();
            indexOuterProduct = indexOuterProduct.flatMap(ai -> {
                return IntStream.range(0, sizes[j]).mapToObj(bi -> append(ai, bi));
            });
        }

        NDarray samplePoints = new NDarray(sizes);
        indexOuterProduct.parallel().forEach(coords -> {
            samplePoints.set(coords, generatingFunction.applyAsDouble(coordsToSamplePoint(coords)));
        });

        return samplePoints;
    }

    private double[] coordsToSamplePoint(int[] coords) {
        double[] point = new double[coords.length];
        for (int i = 0; i < point.length; i++) {
            point[i] = ranges[i].getValue(coords[i]);
        }
        return point;
    }

    @Override
    public double interpolate(double... x) {
        int[] nearestIndices = new int[ranges.length];
        double[] weights = new double[ranges.length];

        for (int i = 0; i < ranges.length; i++) {
            nearestIndices[i] = ranges[i].getNearestIndex(x[i]);
            weights[i] = 1 - ranges[i].getIndexResidualWeight(x[i]);
        }

        return IntStream.range(0, 1 << ranges.length)
                .mapToDouble(weightIndicator -> {
                    double weight = 1;
                    int[] coords = new int[ranges.length];
                    for (int i = 0; i < ranges.length; i++) {
                        int isLowerIdx = (int) (weightIndicator & 0b1);
                        coords[i] = nearestIndices[i] + isLowerIdx;

                        weight *= isLowerIdx == 0 ? weights[i] : 1 - weights[i];
                        weightIndicator = weightIndicator >> 1;
                    }

                    return weight * samplePoints.get(coords);
                }).sum();
    }

    @Override
    public double[] interpolateDerivative(double... x) {
        int[] nearestIndices = new int[ranges.length];
        double[] weights = new double[ranges.length];

        for (int i = 0; i < ranges.length; i++) {
            nearestIndices[i] = ranges[i].getNearestIndex(x[i]);
            weights[i] = 1 - ranges[i].getIndexResidualWeight(x[i]);
        }

        final double[] netWeights = IntStream.range(0, 1 << ranges.length)
                .mapToDouble(weightIndicator -> {
                    double weight = 1;
                    int[] coords = new int[ranges.length];
                    for (int i = 0; i < ranges.length; i++) {
                        int isLowerIdx = (int) (weightIndicator & 0b1);
                        coords[i] = nearestIndices[i] + isLowerIdx;

                        weight *= isLowerIdx == 0 ? weights[i] : 1 - weights[i];
                        weightIndicator = weightIndicator >> 1;
                    }

                    return weight;
                }).toArray();

        return IntStream.range(0, x.length).mapToDouble(dim -> {
            final double dim_weight = weights[dim];
            double derivSum = IntStream.range(0, 1 << ranges.length)
                    .mapToDouble(weightIndicator -> {
                        int[] coords = new int[ranges.length];
                        for (int i = 0; i < ranges.length; i++) {
                            int isLowerIdx = (int) (weightIndicator & 0b1);
                            coords[i] = nearestIndices[i] + isLowerIdx;
                            weightIndicator = weightIndicator >> 1;
                        }

                        double samplePoint = samplePoints.get(coords);
                        int isLowerIdx = (int) (weightIndicator & (0b1 << dim));

                        if (isLowerIdx == 0) {
                            return -samplePoint / dim_weight;
                        } else {
                            return samplePoint / (1 - dim_weight);
                        }
                    }).sum();

            return derivSum
                    / (ranges[dim].getValue(nearestIndices[dim] + 1) - ranges[dim].getValue(nearestIndices[dim]));
        }).toArray();

    }

    private static int[] append(int[] arr, int element) {
        final int N = arr.length;
        arr = Arrays.copyOf(arr, N + 1);
        arr[N] = element;
        return arr;
    }

}
