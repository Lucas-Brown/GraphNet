package com.lucasbrown.NetworkTraining;

/**
 * An abstraction for all classes which interpolate data 
 * 
 * Derivative support is no longer necessary but may be convenient in the future
 */
public interface IInterpolation {

    /**
     * Interpolate the data at the point x.
     * Extrapolation of points may vary in behavior
     * @param x A point to interpolate 
     * @return The interpolated value
     */
    public abstract double interpolate(double... x);

    public abstract double[] interpolateDerivative(double... x);
}
