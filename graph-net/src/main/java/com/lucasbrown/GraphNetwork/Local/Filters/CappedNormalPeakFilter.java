package com.lucasbrown.GraphNetwork.Local.Filters;

import java.util.Random;

public class CappedNormalPeakFilter implements IFilter {

    private final Random rng;

    private double mean, variance, N, minimum;

    public CappedNormalPeakFilter(double mean, double variance, double N, double minimum, Random rng) {
        this.mean = mean;
        this.variance = variance;
        this.N = N;
        this.minimum = minimum;
        this.rng = rng;
    }

    public CappedNormalPeakFilter(double mean, double variance, double minimum, double N) {
        this(mean, variance, N, minimum, new Random());
    }

    public CappedNormalPeakFilter(double mean, double variance, double minimum) {
        this(mean, variance, minimum, 10);
    }

    public double getMean() {
        return mean;
    }

    public double getVariance() {
        return variance;
    }

    public double getMinimumChance(){
        return minimum;
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
        return likelihood(x, mean, variance, minimum);
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
    public void applyAdjustableParameterUpdate(double[] delta) {
        mean -= delta[0];
        variance -= delta[1];
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
    public double[] getLogarithmicParameterDerivative(double x) {
        // ln(this) = -(x-mean)^2/(2*variance^2)
        double w = x - mean;
        double var2 = variance * variance;
        double d_mean = w / var2;
        double d_var = w * w / (var2 * variance);
        double d = getexponentRatios(x);
        return new double[] { d*d_mean, d*d_var };
    }

    @Override
    public double[] getNegatedLogarithmicParameterDerivative(double x) {
        double w = x - mean;
        double var2 = variance * variance;
        double d_mean = w / var2;
        double d_var = w * w / (var2 * variance);
        double d = getNegatedExponentRatios(x);
        return new double[] { d*d_mean, d*d_var };
    }

    public static double likelihood(double x, double mean, double variance, double minimum) {
        double temp = (x - mean) / variance;
        return (1-minimum)*Math.exp(-temp * temp / 2) + minimum;
    }

    public static CappedNormalPeakFilter getStandardNormalFilter() {
        return new CappedNormalPeakFilter(0, 1, 1E-6);
    }

    private double getexponentRatios(double x){
        double w = (x-mean)/(variance);
        double min_ratio = minimum/(1-minimum);
        return 1/(1+min_ratio*Math.exp(w*w/2));
    }

    public double getNegatedExponentRatios(double x) {
        double w = (x-mean)/(variance);
        return 0.99999/(0.99999-Math.exp(w*w/2)); // slightly off for numerical stability
    }

    @Override
    public double getLogarithmicDerivative(double x) {
        double w = (x-mean)/(variance);
        return -getexponentRatios(x) * w/variance;
    }

    @Override
    public double getNegatedLogarithmicDerivative(double x) {
        double w = (x-mean)/(variance);
        return -getNegatedExponentRatios(x)*w/variance;
    }

}
