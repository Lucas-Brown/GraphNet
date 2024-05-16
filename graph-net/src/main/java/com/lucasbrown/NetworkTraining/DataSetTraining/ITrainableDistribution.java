package com.lucasbrown.NetworkTraining.DataSetTraining;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.Convolution.IConvolution;

public interface ITrainableDistribution {
    
    public double getProbabilityDensity(double... x);
    public double getNumberOfPointsInDistribution();
    public double getNormalizationConstant();

    public void applyAdjustments(IExpectationAdjuster adjuster);
    public IConvolution toConvolution(ActivationFunction activator, double weight);

    public IExpectationAdjuster getDefaulAdjuster();


}