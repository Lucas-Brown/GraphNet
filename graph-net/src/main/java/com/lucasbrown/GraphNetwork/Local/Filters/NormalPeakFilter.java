package com.lucasbrown.GraphNetwork.Local.Filters;

import java.util.Random;

import com.lucasbrown.NetworkTraining.DistributionSolverMethods.IExpectationAdjuster;

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
    public void applyAdjustments(IExpectationAdjuster adjuster) {
        double[] updated_params = adjuster.getUpdatedParameters();
        mean = updated_params[0];
        variance = updated_params[1];
        N = updated_params[2];
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
    public void setAdjustableParameters(double[] params) {
        mean = params[0];
        variance = params[1];
    }

    @Override
    public void applyAdjustableParameterUpdate(double[] delta) {
        mean -= delta[0];
        variance -= delta[1];
    }

    @Override
    public double[] getLogarithmicDerivative(double x) {
        // ln(this) = -(x-mean)^2/(2*variance^2)
        double w = x - mean;
        double var2 = variance * variance;
        double d_mean = w / var2;
        double d_var = w * w / (var2 * variance);
        return new double[] { d_mean, d_var };
    }

    @Override
    public double[] getNegatedLogarithmicDerivative(double x) {
        double[] exp_deriv = getLogarithmicDerivative(x);

        double temp = (x - mean) / variance;
        double factor = 0.99999 / (Math.exp(temp * temp / 2) - 0.99999); // set slightly off of 1 for numerical stability

        exp_deriv[0] *= factor;
        exp_deriv[1] *= factor;

        return exp_deriv;
    }

    public static double likelihood(double x, double mean, double variance) {
        double temp = (x - mean) / variance;
        return Math.exp(-temp * temp / 2);
    }

    public static NormalPeakFilter getStandardNormalBetaFilter() {
        return new NormalPeakFilter(0, 1);
    }

}
