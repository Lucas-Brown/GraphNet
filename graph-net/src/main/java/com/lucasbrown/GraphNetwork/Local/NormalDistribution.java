package com.lucasbrown.GraphNetwork.Local;

public class NormalDistribution extends FilterDistribution {

    private double mean;
    private double variance;

    public NormalDistribution(NormalDistribution toCopy)
    {
        this(toCopy.mean, toCopy.variance);
    }

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
    public double sample() {
        return rand.nextGaussian()*variance + mean;
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
    public double[] getParameters() {
        return new double[]{mean, variance};
    }

    @Override
    public void setParameters(double[] params){
        mean = params[0];
        variance = params[1];
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
    
    // fractional error in math formula less than 1.2 * 10 ^ -7.
    // although subject to catastrophic cancellation when z in very close to 0
    // from Chebyshev fitting formula for erf(z) from Numerical Recipes, 6.2
    public static double erf(double z) {
        double t = 1.0 / (1.0 + 0.5 * Math.abs(z));

        // use Horner's method
        double ans = 1 - t * Math.exp( -z*z   -   1.26551223 +
                                            t * ( 1.00002368 +
                                            t * ( 0.37409196 +
                                            t * ( 0.09678418 +
                                            t * (-0.18628806 +
                                            t * ( 0.27886807 +
                                            t * (-1.13520398 +
                                            t * ( 1.48851587 +
                                            t * (-0.82215223 +
                                            t * ( 0.17087277))))))))));
        if (z >= 0) return  ans;
        else        return -ans;
    }

    /**
     * Integral of a gaussian from -infinity to x
     * @param z
     * @return
     */
    public static double phi(double x, double mean, double variance) {
    double z = (x-mean)/(Math.sqrt(2)*variance);
    return (1 + erf(z))/2;
    }

    @Override
    public FilterDistribution copy() {
        return new NormalDistribution(this);
    }

}
