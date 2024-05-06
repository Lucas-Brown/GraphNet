package com.lucasbrown.NetworkTraining.DataSetTraining;

public interface ITrainableDistribution {
    
    public abstract double getProbabilityDensity(double... x);
    public abstract double getNumberOfPointsInDistribution();
    public abstract double getNormalizationConstant();

    public abstract void applyAdjustments(IExpectationAdjuster adjuster);

}
