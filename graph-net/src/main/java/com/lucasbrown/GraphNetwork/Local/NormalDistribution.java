package com.lucasbrown.GraphNetwork.Local;

public class NormalDistribution extends ActivationProbabilityDistribution {

    private double mean;
    private double variance;

    public NormalDistribution(double mean, double variance)
    {
        this.mean = mean;
        this.variance = variance;
    }

    @Override
    public double getProbabilityDensity(double x) {
        double d = (x - mean)/variance;
        return Math.exp(-d*d/2)/Math.sqrt(2*Math.PI)/variance;
    }

    @Override
    public boolean shouldSend(double inputSignal) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'shouldSend'");
    }

    @Override
    public void prepareReinforcement(double valueToReinforce) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'prepareReinforcement'");
    }

    @Override
    public void prepareDiminishment(double valueToDiminish) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'prepareDiminishment'");
    }

    @Override
    public double getMean() {
        return mean;
    }

    @Override
    public double getVariance() {
        return variance;
    }

    @Override
    public void applyAdjustments() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'applyAdjustments'");
    }

    @Override
    public double differenceOfExpectation(double x) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'differenceOfExpectation'");
    }
    
}
