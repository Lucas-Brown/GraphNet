package com.lucasbrown.NetworkTraining.DataSetTraining;

import java.util.Random;

public class NormalPeakFilter implements IFilter{

    private final Random rng;

    private double mean, variance, N;

    public NormalPeakFilter(double mean, double variance, double N, Random rng){
        this.mean = mean;
        this.variance = variance;
        this.N = N;
        this.rng = rng;
    }

    public NormalPeakFilter(double mean, double variance, double N){
        this(mean, variance, N, new Random());
    }

    public NormalPeakFilter(double mean, double variance){
        this(mean, variance, 0);
    }

    public double getMean(){
        return mean;
    }

    public double getVariance(){
        return variance;
    }

    public double getN(){
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

    public static double likelihood(double x, double mean, double variance){
        double w = (x-mean)/variance;
        return Math.exp(-w*w/2);
    }

    public static NormalPeakFilter getStandardNormalBetaFilter()
    {
        return new NormalPeakFilter(0, 1);
    }

    
}
