package com.lucasbrown.NetworkTraining.DistributionSolverMethods;

import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;

public class NoAdjustments implements IExpectationAdjuster  {

    public NoAdjustments(IFilter filter, ITrainableDistribution dist1, ITrainableDistribution dist2){
        
    }

    @Override
    public void prepareAdjustment(double weight, double[] newData) {
        
    }

    @Override
    public void prepareAdjustment(double[] newData) {
        
    }

    @Override
    public void applyAdjustments() {
    }

    @Override
    public double[] getUpdatedParameters() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getUpdatedParameters'");
    }
    
}
