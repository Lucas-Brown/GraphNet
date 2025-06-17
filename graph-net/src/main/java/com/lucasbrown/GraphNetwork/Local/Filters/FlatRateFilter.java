package com.lucasbrown.GraphNetwork.Local.Filters;

import java.util.Random;

/**
 * Allows all signals to pass with the same fixed rate.
 * Adjustments are made in transformed coordinates to prevent full 0% and 100% 
 */
public class FlatRateFilter implements IFilter {

    private final double min = 1E-12;
    private final double max = 1d - min;
    private double rate;
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
    public int getNumberOfAdjustableParameters() {
        return 1;
    }


    @Override
    public double[] getAdjustableParameters() {
        return new double[]{rate};
    }

    @Override
    public void setAdjustableParameters(double... params) {
        rate = params[0];
    }

    
    @Override
    public void setAdjustableParameter(int index, double value) {
        if(index == 0){
            rate = value;
        }
        else
        {
            throw new RuntimeException("Invalid index");
        }
    }

    @Override
    public void applyAdjustableParameterUpdate(double[] delta) {
        rate -= delta[0];
        if(rate > max){
            rate = max;
        }
        else if(rate < min){
            rate = min;
        }
    }

    @Override
    public double[] getLogarithmicParameterDerivative(double x) {
        return new double[]{1/rate};
    }

    @Override
    public double[] getNegatedLogarithmicParameterDerivative(double x) {
        return new double[]{1/(1-rate)};
    }

    @Override
    public double getLogarithmicDerivative(double x) {
        return 0;
    }

    @Override
    public double getNegatedLogarithmicDerivative(double x) {
        return 0;
    }
    
}
