package com.lucasbrown.NetworkTraining.DataSetTraining;

public interface IExpectationAdjuster {
    public abstract void prepareAdjustment(double weight, double... newData);
    public abstract void prepareAdjustment(double... newData);
    public abstract void applyAdjustments();
    public abstract double[] getUpdatedParameters();
}
