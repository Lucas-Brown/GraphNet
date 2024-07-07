package com.lucasbrown.NetworkTraining.DistributionSolverMethods;

import java.util.function.Function;

public interface ITrainableDistribution {
    
    public double getProbabilityDensity(double... x);
    public double getNumberOfPointsInDistribution();
    public double getNormalizationConstant();

    public void applyAdjustments(IExpectationAdjuster adjuster);

    public Function<ITrainableDistribution, IExpectationAdjuster> getDefaulAdjuster();


}
