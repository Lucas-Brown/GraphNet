package com.lucasbrown.GraphNetwork.Distributions;

/**
 * Always lets values pass through
 */
public class OpenFilter extends FilterDistribution {

    @Override
    public FilterDistribution copy() {
        return new OpenFilter();
    }

    @Override
    public double getProbabilityDensity(double x) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getProbabilityDensity'");
    }

    @Override
    public double sample() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'sample'");
    }

    @Override
    public double sendChance(double inputSignal) {
        return 1;
    }

    @Override
    public void prepareReinforcement(double valueToReinforce) {
    }

    @Override
    public void prepareDiminishment(double valueToDiminish) {
    }

    @Override
    public double getMean() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getMean'");
    }

    @Override
    public double getVariance() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getVariance'");
    }

    @Override
    public void applyAdjustments() {
    }

    @Override
    public double differenceOfExpectation(double x) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'differenceOfExpectation'");
    }
    
}
