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

    @Override
    public int getNumberOfAdjustableParameters() {
        return 0;
    }

    @Override
    public double[] getAdjustableParameters() {
        return new double[0];
    }

    @Override
    public void setAdjustableParameters(double[] params) {
        // do nothing
    }

    @Override
    public void applyAdjustableParameterUpdate(double[] delta) {
        // do nothing
    }

    @Override
    public double[] getLogarithmicDerivative(double x) {
        return new double[0];
    }

    @Override
    public double[] getNegatedLogarithmicDerivative(double x) {
        return new double[0];
    }
    
}
