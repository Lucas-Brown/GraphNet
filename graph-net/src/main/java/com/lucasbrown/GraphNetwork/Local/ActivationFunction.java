package com.lucasbrown.GraphNetwork.Local;

/**
 * An activation function allows traditional neural networks to be universal
 * function approximators.
 * In graphs, this is not necessary since all functions can be represented as
 * infinite series which can occur in graphs.
 * Using an activation function may still allow for better results in less time
 * due to their nonlinearity.
 * Activation functions must also be invertible and differentiable for the
 * backwards signal to propagate.
 */
public interface ActivationFunction {

    public abstract double activator(double x);

    public abstract double derivative(double x);

    public abstract double secondDerivative(double x);

    public static final Linear LINEAR = new Linear();
    public static final RectifiedLinearUnit ReLU = new RectifiedLinearUnit();
    // public static final SignedQuadratic SIGNED_QUADRATIC = new SignedQuadratic();
    // public static final SignedLogarithmic SIGNED_LOGARITHMIC = new SignedLogarithmic();

    static class Linear implements ActivationFunction {

        @Override
        public double activator(double x) {
            return x;
        }

        @Override
        public double derivative(double x) {
            return 1;
        }

        @Override
        public double secondDerivative(double x) {
            return 0;
        }

    }

    static class RectifiedLinearUnit implements ActivationFunction {

        @Override
        public double activator(double x) {
            return x < 0 ? 0 : x;
        }

        @Override
        public double derivative(double x) {
            return x < 0 ? 0 : 1;
        }

        @Override
        public double secondDerivative(double x) {
            return 0;
        }

    }

    public static class LeakyRectifiedLinearUnit implements ActivationFunction {

        private final double alpha;

        public LeakyRectifiedLinearUnit(double alpha) {
            this.alpha = alpha;
        }

        @Override
        public double activator(double x) {
            return x < 0 ? x * alpha : x;
        }

        @Override
        public double derivative(double x) {
            return x < 0 ? alpha : 1;
        }

        @Override
        public double secondDerivative(double x) {
            return 0;
        }


    }

    /* 
    static class SignedQuadratic implements ActivationFunction {

        @Override
        public double activator(double x) {
            return Math.signum(x) * x * x / 2 + x;
        }

        @Override
        public double derivative(double x) {
            return Math.abs(x) + 1;
        }

        @Override
        public double inverse(double x) {
            return Math.signum(x) * (Math.sqrt(2 * Math.abs(x) + 1) - 1);
        }

        @Override
        public double inverseDerivative(double x) {
            return 1 / Math.sqrt(2 * Math.abs(x) + 1);
        }

    }

    static class SignedLogarithmic implements ActivationFunction {

        @Override
        public double activator(double x) {
            return Math.signum(x) * Math.log(Math.abs(x) + 1);
        }

        @Override
        public double derivative(double x) {
            return 1 / (Math.abs(x) + 1);
        }

        @Override
        public double inverse(double x) {
            return Math.signum(x) * Math.expm1(Math.abs(x));
        }

        @Override
        public double inverseDerivative(double x) {
            return Math.exp(Math.abs(x));
        }

    }
        */

}