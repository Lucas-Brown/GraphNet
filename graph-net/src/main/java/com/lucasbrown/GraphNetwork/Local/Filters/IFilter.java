package com.lucasbrown.GraphNetwork.Local.Filters;

import com.lucasbrown.NetworkTraining.DistributionSolverMethods.IExpectationAdjuster;

public interface IFilter {
    
    public boolean shouldSend(double x);
    public double getChanceToSend(double x);
    public void applyAdjustments(IExpectationAdjuster adjuster);

    public int getNumberOfAdjustableParameters();
    public double[] getAdjustableParameters();
    public void setAdjustableParameters(double[] params);
    public void applyAdjustableParameterUpdate(double[] delta);

    public double[] getLogarithmicParameterDerivative(double x);
    public double[] getNegatedLogarithmicParameterDerivative(double x);

    public double getLogarithmicDerivative(double x);
    public double getNegatedLogarithmicDerivative(double x);
    
}
