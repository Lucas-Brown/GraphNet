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
    
}