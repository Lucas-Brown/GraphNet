package com.lucasbrown.NetworkTraining.DistributionSolverMethods;

import com.lucasbrown.GraphNetwork.Local.Filters.FlatRateFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;

/**
 * Always allows signals to pass
 */
public class FlatRateFilterBetaAdjuster implements IExpectationAdjuster {

    private double rate;
    private FlatRateFilter filter;
    private BetaDistribution chanceDist;

    public FlatRateFilterBetaAdjuster(IFilter filter, ITrainableDistribution nodeDistribution,
    ITrainableDistribution arcDistribution) {
        this((FlatRateFilter) filter, (BetaDistribution) arcDistribution);
    }

    public FlatRateFilterBetaAdjuster(FlatRateFilter filter, BetaDistribution chanceDist){
        this.filter = filter;
        this.chanceDist = chanceDist;
    }

    @Override
    public void prepareAdjustment(double weight, double[] newData) {

    }

    @Override
    public void prepareAdjustment(double[] newData) {
    }

    @Override
    public void applyAdjustments() {
        rate = chanceDist.getMean();
        filter.applyAdjustments(this);
    }

    @Override
    public double[] getUpdatedParameters() {
        return new double[]{rate};
    }

}
