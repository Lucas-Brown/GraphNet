package com.lucasbrown.GraphNetwork.Local.Filters;

import java.util.Random;

public class NormalPeakFilter implements IFilter {

    private final Random rng;

    private double mean, variance, N;

    public NormalPeakFilter(double mean, double variance, double N, Random rng) {
        this.mean = mean;
        this.variance = variance;
        this.N = N;
        this.rng = rng;
    }

    public NormalPeakFilter(double mean, double variance, double N) {
        this(mean, variance, N, new Random());
    }

    public NormalPeakFilter(double mean, double variance) {
        this(mean, variance, 10);
    }

    public double getMean() {
        return mean;
    }

    public double getVariance() {
        return variance;
    }

    public double getN() {
        return N;
    }

    @Override
    public boolean shouldSend(double x) {
        return getChanceToSend(x) < rng.nextDouble();
    }

    @Override
    public double getChanceToSend(double x) {
        return likelihood(x, mean, variance);
    }

    @Override
    public int getNumberOfAdjustableParameters() {
        return 2;
    }

    @Override
    public double[] getAdjustableParameters() {
        return new double[] { mean, variance };
    }

    @Override
    public void setAdjustableParameters(double... params) {
        mean = params[0];
        variance = params[1];
    }

    @Override
    public void setAdjustableParameter(int index, double value)
    {
        switch(index)
        {
            case 0:
                mean = value;
                break;
            case 1:
                variance = value;
                break;
            default:
                throw new RuntimeException("Invalid index");
        }

    }

    @Override
    public void applyAdjustableParameterUpdate(double[] delta) {
        mean -= delta[0];
        variance -= delta[1];
    }

    @Override
    public double[] getLogarithmicParameterDerivative(double x) {
        // ln(this) = -(x-mean)^2/(2*variance^2)
        double w = x - mean;
        double var2 = variance * variance;
        double d_mean = w / var2;
        double d_var = w * w / (var2 * variance);
        return new double[] { d_mean, d_var };
    }

    @Override
    public double[] getNegatedLogarithmicParameterDerivative(double x) {
        double[] exp_deriv = getLogarithmicParameterDerivative(x);
        final double stabilityFactor = 1-1E-12;

        double temp = (x - mean) / variance;
        double factor = stabilityFactor / (Math.exp(temp * temp / 2) - stabilityFactor); // set slightly off of 1 for numerical stability

        exp_deriv[0] *= factor;
        exp_deriv[1] *= factor;

        return exp_deriv;
    }


    @Override
    public double getLogarithmicDerivative(double x) {
        return (x-mean)/(variance*variance);
    }

    @Override
    public double getNegatedLogarithmicDerivative(double x) {
        double normal_derivative = getLogarithmicDerivative(x);
        double likelihood = getChanceToSend(x);
        return -likelihood/(1-likelihood) * normal_derivative;
    }
    
    public static double likelihood(double x, double mean, double variance) {
        double temp = (x - mean) / variance;
        return Math.exp(-temp * temp / 2);
    }

    public static NormalPeakFilter getStandardNormalBetaFilter() {
        return new NormalPeakFilter(0, 1);
    }

}
