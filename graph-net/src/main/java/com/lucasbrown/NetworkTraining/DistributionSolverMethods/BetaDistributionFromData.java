package com.lucasbrown.NetworkTraining.DistributionSolverMethods;

public class BetaDistributionFromData extends BetaDistributionAdjuster {

    public BetaDistributionFromData(ITrainableDistribution distribution) {
        super((BetaDistribution) distribution);
    }

    public BetaDistributionFromData(BetaDistribution distribution) {
        super(distribution);
    }

    protected double logLikelihoodExpectationOfDistribution(double lambda_alpha, double lambda_beta) {
        return 0;
    }

}
