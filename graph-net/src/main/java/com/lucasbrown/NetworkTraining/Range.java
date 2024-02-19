package com.lucasbrown.NetworkTraining;

import java.util.Arrays;
import java.util.stream.DoubleStream;

/**
 * A range is a collection of values which is ordered and unique.
 * If a range is generated from a known function, Ranges can often be searched
 * in constant-time using the inverse of the generating function.
 */
public abstract class Range {

    protected final int n_divisions;
    protected double[] values;

    public Range(int n_divisions) {
        this.n_divisions = n_divisions;
        values = new double[n_divisions];
    }

    /**
     * Check whether the provided value is within the range
     * 
     * @param x
     * @return
     */
    public boolean isWithinRange(double x) {
        return x >= values[0] && x <= values[values.length - 1];
    }

    public int getNumberOfPoints() {
        return values.length;
    }

    /**
     * Whether the index is within the range
     * 
     * @param index
     * @return
     */
    public boolean isValidIndex(int index) {
        return index >= 0 && index < values.length;
    }

    /**
     * Get the value at the index
     * 
     * @param index
     * @return
     */
    public double getValue(int index) {
        return values[index];
    }

    public DoubleStream stream() {
        return Arrays.stream(values);
    }

    /**
     * Get the index of the lower
     * 
     * @param x
     * @return
     */

    public int getNearestIndex(double x) {
        double floatingIndex = getFloatingIndex(x);
        if (floatingIndex < 0 || floatingIndex >= n_divisions) {
            return -2;
        } else {
            int index = (int) floatingIndex;
            if (floatingIndex % 1d == 0) {
                return index - 1;
            } else {
                return index;
            }
        }
    }

    /**
     * Returns how close x is to the nearest value less than x
     * Different range mappings may choose to weigh how close a value is to the next
     * nearest index.
     * For example, let x = 1.5 with it's nearest neighbors 1 and 2.
     * a linear mapping would return 0.5
     * a non linear mapping (such as 2^x -1) may return 0.41
     * 
     * @param x
     * @return
     */
    public double getIndexResidualWeight(double x) {
        double floatingIndex = getFloatingIndex(x);
        return Math.ceil(floatingIndex) - floatingIndex;
    }

    protected abstract double getFloatingIndex(double x);

}
