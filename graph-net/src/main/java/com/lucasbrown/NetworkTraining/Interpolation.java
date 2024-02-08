package com.lucasbrown.NetworkTraining;

public interface Interpolation {

    public abstract double interpolate(double... x);
    public abstract double[] interpolateDerivative(double... x);
}
