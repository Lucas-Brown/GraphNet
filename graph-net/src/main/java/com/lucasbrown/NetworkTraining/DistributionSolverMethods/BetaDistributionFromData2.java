package com.lucasbrown.NetworkTraining.DistributionSolverMethods;

import java.security.InvalidParameterException;
import java.util.ArrayList;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.HelperClasses.WeightedDouble;

public class BetaDistributionFromData2 implements IExpectationAdjuster {

    private double alpha, beta, N;

    private BetaDistribution distribution;

    private ArrayList<WeightedDouble> newPoints;

    public BetaDistributionFromData2(ITrainableDistribution distribution) {
        this((BetaDistribution) distribution);
    }

    public BetaDistributionFromData2(BetaDistribution distribution) {
        this.distribution = distribution;
        newPoints = new ArrayList<>();
    }

    public double getAlpha() {
        return alpha;
    }

    public double getBeta() {
        return beta;
    }

    public double getN() {
        return N;
    }

    public void prepareAdjustment(double weight, double value) {
        if (value < 0 || value > 1) {
            throw new InvalidParameterException("The beta distribution is bounded between 0 and 1");
        }
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
        double N_sample = WeightedDouble.getWeightSum(newPoints);

        if (N_sample == 0) {
            newPoints.clear();
            return; // do not change
        }

        N = N_sample;
        newPoints.forEach(wd -> wd.value = BetaDistribution.betaDataTransformation(wd.value, N_sample));

        double mean = WeightedDouble.getWeightedMean(newPoints, N);
        double variance = WeightedDouble.getWeightedVariance(newPoints, N, mean);

        // parameter estimation using mean and variance
        double temp = mean * (1 - mean) / variance - 1;
        alpha = mean * temp;
        beta = (1 - mean) * temp;

        N = Math.min(N, GraphNetwork.N_MAX);

        newPoints.clear();
        distribution.applyAdjustments(this);
    }

    @Override
    public double[] getUpdatedParameters() {
        return new double[] { alpha, beta, N };
    }

}
