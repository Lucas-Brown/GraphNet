package com.lucasbrown.NetworkTraining.DataSetTraining;

import com.lucasbrown.NetworkTraining.ApproximationTools.Convolution.IConvolution;

public interface IFilter {
    
    public boolean shouldSend(double x);
    public double getChanceToSend(double x);
    public void applyAdjustments(IExpectationAdjuster adjuster);
}
