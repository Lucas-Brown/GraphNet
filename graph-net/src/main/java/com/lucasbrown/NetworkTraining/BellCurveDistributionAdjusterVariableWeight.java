package com.lucasbrown.NetworkTraining;

import com.lucasbrown.GraphNetwork.Local.BellCurveDistribution;

/**
 * Testing of this class showed that it is an improper representation and should not be used
 */
public class BellCurveDistributionAdjusterVariableWeight extends BellCurveDistributionAdjuster {

    public BellCurveDistributionAdjusterVariableWeight(BellCurveDistribution parentDistribution, boolean use_map) {
        super(parentDistribution, use_map);
    }

    public BellCurveDistributionAdjusterVariableWeight(BellCurveDistribution parentDistribution) {
        super(parentDistribution);
    }


    /**
     * The log-likelihood for a single data point with position x and reinforcement
     * value b
     * 
     * @param x        position
     * @param b        reinforcement state (true for reinforcment, false for
     *                 diminishment)
     * @param mean
     * @param variance
     * @return
     */
    @Override
    public double logLikelihood(double x, boolean b, double mean, double variance) {
        if (x == mean)
            return 0;

        final double rate = -(x - mean) * (x - mean) / (2 * variance * variance);
        final double rate_exp = Math.exp(rate);
        final double omega = -Math.log(2 * Math.PI * variance * variance) / 2;
        if (b) {
            return (2 * rate + omega)/rate_exp;
        } else {
            return (Math.log(1 - rate_exp) + rate + omega)/(1-rate_exp);
        }
    }
}
