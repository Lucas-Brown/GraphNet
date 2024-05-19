package com.lucasbrown.NetworkTraining.DataSetTraining;

/**
 * Always allows signals to pass
 */
public class OpenFilter implements IFilter {

    @Override
    public boolean shouldSend(double x) {
        return true;
    }

    @Override
    public double getChanceToSend(double x) {
        return 1;
    }

    @Override
    public void applyAdjustments(IExpectationAdjuster adjuster) {
        // Do nothing
    }
    
}
