package com.lucasbrown.NetworkTraining.OutputDerivatives;

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
            double abs_delta = Math.abs(value - target);
            return (abs_delta + 1) * Math.log(abs_delta + 1) - abs_delta;
        }

        @Override
        public double error_derivative(double value, double target) {
            double delta = value - target;
            return Math.signum(delta) * Math.log(Math.abs(delta) + 1);
        }

        @Override
        public double error_second_derivative(double value, double target) {
            return 1 / (Math.abs(value - target) + 1);
        }

    }

    /**
     * mean squared error scaled by the target value
     */
    public static class AugmentedRelativeError implements ErrorFunction {

        public double eps = 1E-12;

        @Override
        public double error(double value, double target) {
            double delta = value - target;
            return delta * delta / 2 / (target * target + eps);
        }

        @Override
        public double error_derivative(double value, double target) {
            return (value - target) / (target * target + eps);
        }

        @Override
        public double error_second_derivative(double value, double target) {
            return 1 / (target * target + eps);
        }

    }

    /**
     * cross-entropy loss/logarithmic loss
     */
    public static class CrossEntropy implements ErrorFunction {

        public double eps = 1E-12;

        @Override
        public double error(double value, double target) {
            if (target == 1) {
                return -Math.log(value + eps);
            } else if (target == 0) {
                return -Math.log(1 + eps - value);
            } else {
                return -(target * Math.log(value + eps) + (1 - target) * Math.log(1 + eps - value));
            }
        }

        @Override
        public double error_derivative(double value, double target) {
            if (target == 1) {
                return -1 / (value + eps);
            } else if (target == 0) {
                return 1 / (1 + eps - value);
            } else {
                return (1 - target) / (1 + eps - value) - target / (value + eps);
            }
        }

        @Override
        public double error_second_derivative(double value, double target) {
            if (target == 1) {
                return 1 / ((value + eps) * (value + eps));
            } else if (target == 0) {
                return 1 / ((1 + eps - value) * (1 + eps - value));
            } else {
                return target / ((value + eps) * (value + eps)) + ((1 + eps - value) * (1 + eps - value));
            }
        }

    }

}
