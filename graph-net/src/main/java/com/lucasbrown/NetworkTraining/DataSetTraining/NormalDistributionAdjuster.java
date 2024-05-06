package com.lucasbrown.NetworkTraining.DataSetTraining;

import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.List;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.optimization.NelderMead;

public class NormalDistributionAdjuster implements IExpectationAdjuster {

    private double mean, variance, N;

    private NormalDistribution distribution;

    private ArrayList<WeightedDouble> newPoints;

    public NormalDistributionAdjuster(NormalDistribution distribution) {
        this.distribution = distribution;
    }

    public double getMean(){
        return mean;
    }

    public double getVariance(){
        return variance;
    }

    public void prepareAdjustment(double weight, double value) {
        if (value < 0 || value > 1) {
            throw new InvalidParameterException("The beta distribution is bounded between 0 and 1");
        }
        newPoints.add(new WeightedDouble(weight, value));
    }

    @Override
    public void prepareAdjustment(double weight, double... newPoint) {
        if(newPoint.length > 1){
            throw new InvalidParameterException("This distribution only accepts a single degree of input. ");
        }
        prepareAdjustment(weight, newPoint[0]);
    }

    @Override
    public void prepareAdjustment(double... newData) {
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

        mean = mean_updated;
        N = N_updated;
        newPoints.clear();
    }


    @Override
    public double[] getUpdatedParameters() {
        return new double[]{mean, variance};
    }

}
