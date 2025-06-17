package com.lucasbrown.GraphNetwork.Local.Filters;

public interface IFilter {
    
    public boolean shouldSend(double x);
    public double getChanceToSend(double x);

    public int getNumberOfAdjustableParameters();
    public double[] getAdjustableParameters();
    public void setAdjustableParameters(double... params);
    public void setAdjustableParameter(int index, double value);
    public void applyAdjustableParameterUpdate(double[] delta);

    public double[] getLogarithmicParameterDerivative(double x);
    public double[] getNegatedLogarithmicParameterDerivative(double x);

    public double getLogarithmicDerivative(double x);
    public double getNegatedLogarithmicDerivative(double x);
    
}
