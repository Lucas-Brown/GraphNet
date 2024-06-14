package com.lucasbrown.NetworkTraining.DataSetTraining;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;

public class BernoulliDistributionAdjuster extends WeightedDoubleCollector {

    private BernoulliDistribution distribution;
    private double p, N;

    public BernoulliDistributionAdjuster(ITrainableDistribution distribution) {
        this((BernoulliDistribution) distribution);
    }

    public BernoulliDistributionAdjuster(BernoulliDistribution distribution) {
        this.distribution = distribution;
    }

    @Override
    public void applyAdjustments() {
        p = distribution.getP();
        N = distribution.getNumberOfPointsInDistribution();

        double N_new = WeightedDouble.getWeightSum(newPoints);
        if(N_new == 0){
            newPoints.clear();
            return;
        }
        double p_new = WeightedDouble.getWeightedMean(newPoints, N_new);

        p = (p * N + p_new * N_new) / (N + N_new);
        N = Math.min(N + N_new, GraphNetwork.N_MAX);
        distribution.applyAdjustments(this);
        newPoints.clear();
    }

    @Override
    public double[] getUpdatedParameters() {
        return new double[] { p, N };
    }

}
