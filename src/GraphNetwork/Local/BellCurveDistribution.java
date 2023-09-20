package src.GraphNetwork.Local;

import java.util.Random;

/**
 * A bell curve distribution 
 */
public class BellCurveDistribution extends ActivationProbabilityDistribution {

    /**
     * Random number generator for probabalistically choosing whether to send a signal
     */
    private Random rand = new Random();

    /**
     * mean value and standard deviation of a normal distribution
     */
    private double mean, standardDeviation;

    private int N;

    /**
     * @param mean mean value of the normal distribution
     * @param standardDeviation standard deviation of the normal distribution
     */
    public BellCurveDistribution(double mean, double standardDeviation, double strength)
    {
        this.mean = mean;
        this.standardDeviation = standardDeviation;
        this.strength = strength;
        N = 100;
    }

    /**
     * normalized normal distribution
     * @param x 
     * @return value of the distribution at the point x. always returns on the interval (0, 1]
     */
    private double computeNormalizedDist(double x)
    {
        final double temp = (x-mean)/standardDeviation;
        return (double) Math.exp(-temp*temp/2);
    }

    /**
     * Compute the updated mean and standard deviation using a fixed-count approximation.
     * @param newPoint The new data to update the distribution 
     * @param N The fixed number of data points in the distribution 
     */
    private void updateMeanAndVariance(double newPoint, int N_Limiter)
    {
        // Useful constants
        final double Np1Inv = 1f/(N + 1f);
        final double distanceFromMean = newPoint - mean;

        // Compute the updated variance (standard deviation)
        standardDeviation = standardDeviation * standardDeviation + distanceFromMean*distanceFromMean*Np1Inv;
        standardDeviation = (double) Math.sqrt(N*Np1Inv * (standardDeviation));

        // Compute the new mean value 
        mean = (N*mean + newPoint)*Np1Inv;   

        if(N <= N_Limiter)
        {
            N++;
        } 
    }

    @Override
    public boolean shouldSend(double inputSignal, double factor) {
        // Use the normalized normal distribution as a measure of how likely  
        return factor*computeNormalizedDist(inputSignal) >= rand.nextDouble();
    }

    @Override
    public double getOutputStrength() {
        return strength;
    }

    @Override
    protected void updateDistribution(double backpropSignal, int N_Limiter) {
        // Update the distribution mean and variance
        updateMeanAndVariance(backpropSignal, N_Limiter);

    }

    @Override
    public double getMostLikelyValue()
    {
        return mean;
    }
    
}
