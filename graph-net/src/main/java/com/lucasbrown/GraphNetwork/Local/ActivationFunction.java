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

    public abstract double inverse(double x);

    public abstract double inverseDerivative(double x);

    public static final Linear LINEAR = new Linear();
    public static final SignedQuadratic SIGNED_QUADRATIC = new SignedQuadratic();

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
        public double inverse(double x) {
            return x;
        }

        @Override
        public double inverseDerivative(double x) {
            return 1;
        }

    }

    static class SignedQuadratic implements ActivationFunction {

        @Override
        public double activator(double x) {
            return Math.signum(x) * x * x/2 + x;
        }

        @Override
        public double derivative(double x) {
            return Math.abs(x) + 1;
        }

        @Override
        public double inverse(double x) {
            return Math.signum(x) * (Math.sqrt(2*Math.abs(x) + 1) - 1);
        }

        @Override
        public double inverseDerivative(double x) {
            return 1 / Math.sqrt(2*Math.abs(x) + 1);
        }

    }

}