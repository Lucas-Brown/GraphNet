package com.lucasbrown.NetworkTraining.DistributionSolverMethods;

import java.security.InvalidParameterException;
import java.util.ArrayList;

import com.lucasbrown.HelperClasses.WeightedDouble;

public class NormalDistributionFromData implements IExpectationAdjuster {

    private double mean, variance, N;

    private NormalDistribution distribution;

    private ArrayList<WeightedDouble> newPoints;

    public NormalDistributionFromData(ITrainableDistribution distribution) {
        this((NormalDistribution) distribution);
    }

    public NormalDistributionFromData(NormalDistribution distribution) {
        this.distribution = distribution;
        newPoints = new ArrayList<WeightedDouble>();
    }

    public double getMean() {
        return mean;
    }

    public double getVariance() {
        return variance;
    }

    public void prepareAdjustment(double weight, double value) {
        newPoints.add(new WeightedDouble(weight, value));
    }

    @Override
    public void prepareAdjustment(double weight, double[] newPoint) {
        if (newPoint.length > 1) {
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

        mean = 0;
        N = 0;
        for (WeightedDouble wd : newPoints) {
            mean += wd.weight * wd.value;
            N += wd.weight;
        }
        mean /= N;

        variance = Math.sqrt(
                newPoints.stream().mapToDouble(wd -> wd.weight * Math.pow(wd.value - mean, 2)).sum() / N);

        distribution.applyAdjustments(this);
    }

    @Override
    public double[] getUpdatedParameters() {
        return new double[] { mean, variance, N };
    }

}
