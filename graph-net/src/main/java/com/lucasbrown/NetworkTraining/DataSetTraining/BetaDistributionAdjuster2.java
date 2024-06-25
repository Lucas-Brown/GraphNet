package com.lucasbrown.NetworkTraining.DataSetTraining;

import java.security.InvalidParameterException;
import java.util.ArrayList;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;

public class BetaDistributionAdjuster2 implements IExpectationAdjuster {

    private final double alpha_prior, beta_prior, v_prior;

    private double alpha, beta, N;

    private BetaDistribution distribution;

    private ArrayList<WeightedDouble> newPoints;

    public BetaDistributionAdjuster2(ITrainableDistribution distribution) {
        this((BetaDistribution) distribution);
    }

    public BetaDistributionAdjuster2(BetaDistribution distribution) {
        this.distribution = distribution;
        newPoints = new ArrayList<>();
        alpha_prior = distribution.getAlpha();
        beta_prior = distribution.getBeta();
        v_prior = alpha_prior + beta_prior;
    }

    public double getAlpha(){
        return alpha;
    }

    public double getBeta(){
        return beta;
    }

    public double getN(){
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

        double N_points = WeightedDouble.getWeightSum(newPoints);

        if(N_points == 0){
            newPoints.clear();
            return; // make no changes
        }

        N = distribution.getNumberOfPointsInDistribution();

        double mu0 = alpha_prior / v_prior;
        double var2 = alpha_prior * beta_prior / (v_prior * v_prior * (v_prior + 1));
        double N_new = N + N_points;

        newPoints.forEach(wd -> wd.value = BetaDistribution.betaDataTransformation(wd.value, N_points));
        

        double sample_mean = WeightedDouble.getWeightedMean(newPoints, N_points);
        double sample_variance = WeightedDouble.getWeightedVariance(newPoints, N_points, sample_mean);

        double mean = (N*mu0 + N_points*sample_mean)/N_new;
        double variance = (N*var2 + N_points*sample_variance)/N_new;

        // parameter estimation using mean and variance
        double temp = mean*(1-mean)/variance - 1;
        alpha = mean*temp;
        beta = (1-mean)*temp;

        N = Math.min(N_new, GraphNetwork.N_MAX);

        assert Double.isFinite(alpha) && alpha > 0;
        assert Double.isFinite(beta) && beta > 0;
        newPoints.clear();
        distribution.applyAdjustments(this);
    }

    @Override
    public double[] getUpdatedParameters() {
        return new double[]{alpha, beta, N};
    }

}
