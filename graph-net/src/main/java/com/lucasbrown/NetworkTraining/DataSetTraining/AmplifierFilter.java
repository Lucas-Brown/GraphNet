package com.lucasbrown.NetworkTraining.DataSetTraining;

import java.util.Random;

/**
 * Always allows signals to pass
 */
public class AmplifierFilter implements IFilter {

    private double maxAmplification;
    private double alpha;
    private double rate;
    private Random rng;

    public AmplifierFilter(double rate, double maxAmplification){
        this.maxAmplification = maxAmplification;
        this.rate = rate;
        this.alpha = rateToAlpha(rate);
        rng = new Random();
    }

    @Override
    public boolean shouldSend(double x) {
        return rate >= 1 ? true : rng.nextDouble() <= rate;
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
        return maxAmplification/(1+Math.exp(-x));
    } 

    private double rateToAlpha(double x){
        return Math.log(x/(maxAmplification-x));
    }

    @Override
    public double[] getAdjustableParameters() {
        return new double[]{rateToAlpha(rate)};
    }

    @Override
    public void setAdjustableParameters(double[] params) {
        alpha = params[0];
        rate = alphaToRate(params[0]);
    }

    @Override
    public void applyAdjustableParameterUpdate(double[] delta) {
        alpha -= delta[0];
        rate = alphaToRate(alpha);
    }

    @Override
    public double[] getLogarithmicDerivative(double x) {
        return new double[]{1-rate/maxAmplification};
    }

    @Override
    public double[] getNegatedLogarithmicDerivative(double x) {
        double part1 = 1/(1-maxAmplification + Math.exp(-alpha));
        double part2 = -1/(1-rate);
        return new double[]{rate*(part1 + part2)};
    }
    
}
