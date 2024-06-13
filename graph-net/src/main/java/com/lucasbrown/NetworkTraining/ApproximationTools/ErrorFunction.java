package com.lucasbrown.NetworkTraining.ApproximationTools;

public interface ErrorFunction {

    public abstract double error(double value, double target);

    public abstract double error_derivative(double value, double target);

    public abstract double error_second_derivative(double value, double target);

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
        
        @Override
        public double error_second_derivative(double value, double target) {
            return 1;
        }

    }
    
    /**
     * logarithmic derivative
     */
    public static class LogarithmicError implements ErrorFunction {

        @Override
        public double error(double value, double target) {
            double abs_delta = Math.abs(value-target);
            return (abs_delta + 1)*Math.log(abs_delta+1) - abs_delta;
        }

        @Override
        public double error_derivative(double value, double target) {
            double delta = value-target;
            return Math.signum(delta)*Math.log(Math.abs(delta) + 1);
        }
        
        @Override
        public double error_second_derivative(double value, double target) {
            return 1/(Math.abs(value-target) + 1);
        }

    }

}
