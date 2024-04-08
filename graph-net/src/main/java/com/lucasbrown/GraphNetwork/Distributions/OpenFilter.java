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
        return 0;
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
    public void prepareReinforcement(double valueToReinforce, double weight) {
    }

    @Override
    public void prepareDiminishment(double valueToDiminish, double weight) {
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
    public double getNumberOfPointsInDistribution()
    {
        return 1;
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
