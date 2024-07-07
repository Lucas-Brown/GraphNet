package com.lucasbrown.NetworkTraining.DistributionSolverMethods;

public interface IExpectationAdjuster {
    public abstract void prepareAdjustment(double weight, double[] newData);
    public abstract void prepareAdjustment(double[] newData);
    public abstract void applyAdjustments();
    public abstract double[] getUpdatedParameters();
}
