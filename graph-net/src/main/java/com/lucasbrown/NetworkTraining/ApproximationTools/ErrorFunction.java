package com.lucasbrown.NetworkTraining.ApproximationTools;

public interface ErrorFunction {

    public abstract double error(double value, double target);

    public abstract double error_derivative(double value, double target);

    /**
     * mean squared error
     */
    public static class MeanSquaredError implements ErrorFunction {

        @Override
        public double error(double value, double target) {
            return (value - target) * (value - target) / 2;
        }

        @Override
        public double error_derivative(double value, double target) {
            return (value - target);
        }

    }

}
