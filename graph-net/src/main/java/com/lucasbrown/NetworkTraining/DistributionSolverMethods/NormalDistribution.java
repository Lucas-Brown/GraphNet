package com.lucasbrown.NetworkTraining.DistributionSolverMethods;

import java.util.Random;
import java.util.function.Function;

public class NormalDistribution implements ITrainableDistribution{

    private Random random;
    private double mean, variance, N;

    public NormalDistribution(double mean, double variance, double N, Random random){
        this.mean = mean;
        this.variance = variance;
        this.N = N;
    }

    public NormalDistribution(double mean, double variance, double N){
        this(mean, variance, N, new Random());
    }
    
    public NormalDistribution(double mean, double variance){
        this(mean, variance, 10);
    }

    @Override
    public double getProbabilityDensity(double... x) {
        return getProbabilityDensity(x[0]);
    }

    public double getProbabilityDensity(double x){
        return densityOfPoint(x, mean, variance);
    }

    @Override 
    public double getNumberOfPointsInDistribution(){
        return N;
    }

    public double getMean() {
        return mean;
    }

    public double getVariance(){
        return variance;
    }

    @Override
    public double getNormalizationConstant(){
        return normalizationConstant(variance);
    }
    
    @Override
    public void applyAdjustments(IExpectationAdjuster adjuster) {
        double[] params = adjuster.getUpdatedParameters();
        mean = params[0];
        variance = params[1];
        N = params[2];
    }

    public double sample(){
        return random.nextGaussian()*variance + mean;
    }

    @Override
    public Function<ITrainableDistribution, IExpectationAdjuster> getDefaulAdjuster() {
        return NormalDistributionAdjuster::new;
    }


    public static double normalizationConstant(double variance){
        return Math.sqrt(2*Math.PI*variance*variance);
    }

    public static double densityOfPoint(double x, double mean, double variance){
        double u = (x - mean) / variance;
        return Math.exp(-u*u/2)/normalizationConstant(variance);
    }

    public static NormalDistribution getStandardNormalDistribution()
    {
        return new NormalDistribution(0, 1);
    }

}
