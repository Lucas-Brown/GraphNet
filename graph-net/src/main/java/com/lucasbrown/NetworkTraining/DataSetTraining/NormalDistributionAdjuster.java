package com.lucasbrown.NetworkTraining.DataSetTraining;

import java.security.InvalidParameterException;
import java.util.ArrayList;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;

public class NormalDistributionAdjuster implements IExpectationAdjuster {

    private double mean, variance, N;

    private NormalDistribution distribution;

    private ArrayList<WeightedDouble> newPoints;

    public NormalDistributionAdjuster(ITrainableDistribution distribution) {
        this((NormalDistribution) distribution);
    }

    public NormalDistributionAdjuster(NormalDistribution distribution) {
        this.distribution = distribution;
        newPoints = new ArrayList<WeightedDouble>();
    }

    public double getMean(){
        return mean;
    }

    public double getVariance(){
        return variance;
    }

    public void prepareAdjustment(double weight, double value) {
        newPoints.add(new WeightedDouble(weight, value));
    }

    @Override
    public void prepareAdjustment(double weight, double[] newPoint) {
        if(newPoint.length > 1){
            throw new InvalidParameterException("This distribution only accepts a single degree of input. ");
        }
        prepareAdjustment(weight, newPoint[0]);
    }

    @Override
    public void prepareAdjustment(double[] newData) {
        prepareAdjustment(1, newData);
    }

    @Override
    public void applyAdjustments() {
        mean = distribution.getMean();
        variance = distribution.getVariance();
        N = distribution.getNumberOfPointsInDistribution();

        double W = newPoints.stream().mapToDouble(wp -> wp.weight).sum();
        double N_updated = N + W;

        double mean_shift = newPoints.stream().mapToDouble(wp -> wp.weight*wp.value).sum();
        mean_shift = (mean_shift - N*mean)/N_updated;
        double mean_updated = mean + mean_shift;

        variance *= variance;
        variance += mean_shift*mean_shift;
        variance *= N;
        variance += newPoints.stream().mapToDouble(wp -> wp.weight*(wp.value - mean_updated)*(wp.value - mean_updated)).sum();
        variance /= N_updated;
        variance = Math.sqrt(variance);

        mean = mean_updated;
        N = Math.min(N_updated, GraphNetwork.N_MAX);
        newPoints.clear();
        distribution.applyAdjustments(this);
    }


    @Override
    public double[] getUpdatedParameters() {
        return new double[]{mean, variance, N};
    }

}
