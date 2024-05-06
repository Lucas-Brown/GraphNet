package com.lucasbrown.NetworkTraining.DataSetTraining;

public interface IFilter {
    
    public boolean shouldSend(double x);
    public double getChanceToSend(double x);
    public void applyAdjustments(IExpectationAdjuster adjuster);

}
