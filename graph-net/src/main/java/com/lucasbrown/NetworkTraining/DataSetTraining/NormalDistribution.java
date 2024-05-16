package com.lucasbrown.NetworkTraining.DataSetTraining;

import java.util.Random;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.Convolution.IConvolution;
import com.lucasbrown.NetworkTraining.ApproximationTools.Convolution.LinearNormalConvolution;

public class NormalDistribution extends BackwardsSamplingDistribution{

    private double mean, variance, N;

    public NormalDistribution(double mean, double variance, double N, Random random){
        super(random);
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
    }


    @Override
    public IConvolution toConvolution(ActivationFunction activator, double weight) {
        if(activator.equals(ActivationFunction.LINEAR)){
            return new LinearNormalConvolution(this, weight);
        }

        return super.toConvolution(activator, weight);
    }

    @Override
    public double sample(){
        return rng.nextGaussian()*variance + mean;
    }

    public static double normalizationConstant(double variance){
        return Math.sqrt(2*Math.PI*variance*variance);
    }

    public static double densityOfPoint(double x, double mean, double variance){
        double u = (x - mean) / variance;
        return Math.exp(-u*u/2)/normalizationConstant(variance);
    }

    @Override
    public IExpectationAdjuster getDefaulAdjuster() {
        return new NormalDistributionAdjuster(this);
    }


}
