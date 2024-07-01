package com.lucasbrown.NetworkTraining.DataSetTraining;

import java.util.Random;

/**
 * Always allows signals to pass
 */
public class FlatRateFilter implements IFilter {

    double rate;
    private Random rng;

    public FlatRateFilter(double rate){
        this.rate = rate;
        rng = new Random();
    }

    @Override
    public boolean shouldSend(double x) {
        return rng.nextDouble() <= rate;
    }

    @Override
    public double getChanceToSend(double x) {
        return rate;
    }

    @Override
    public void applyAdjustments(IExpectationAdjuster adjuster) {
        rate = adjuster.getUpdatedParameters()[0];
    }

    @Override
    public int getNumberOfAdjustableParameters() {
        return 1;
    }

    private double alphaToRate(double x){
        return 1/(1+Math.exp(-x));
    } 

    private double rateToAlpha(double x){
        return Math.log(x/(1-x));
    }

    @Override
    public double[] getAdjustableParameters() {
        return new double[]{rateToAlpha(rate)};
    }

    @Override
    public void setAdjustableParameters(double[] params) {
        rate = alphaToRate(params[0]);
    }

    @Override
    public void applyAdjustableParameterUpdate(double[] delta) {
        rate = alphaToRate(rateToAlpha(rate) - delta[0]);
        assert rate >= 0 && rate <= 1;
    }

    @Override
    public double[] getLogarithmicDerivative(double x) {
        return new double[]{1-rate};
    }

    @Override
    public double[] getNegatedLogarithmicDerivative(double x) {
        return new double[]{-rate};
    }
    
}
