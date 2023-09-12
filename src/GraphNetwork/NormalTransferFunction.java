package src.GraphNetwork;

import java.util.Random;

public class NormalTransferFunction extends NodeTransferFunction {

    /**
     * Random number generator for probabalistically choosing whether to send a signal
     */
    private Random rand = new Random();

    /**
     * mean value and standard deviation of a normal distribution
     */
    private float mean, standardDeviation;

    private int N;

    /**
     * @param mean mean value of the normal distribution
     * @param standardDeviation standard deviation of the normal distribution
     */
    public NormalTransferFunction(float mean, float standardDeviation, float strength)
    {
        this.mean = mean;
        this.standardDeviation = standardDeviation;
        this.strength = strength;
        N = 1;
    }

    /**
     * normalized normal distribution
     * @param x 
     * @return value of the distribution at the point x. always returns on the interval (0, 1]
     */
    private float ComputeNormalizedDist(float x)
    {
        final float temp = (x-mean)/standardDeviation;
        return (float) Math.exp(-temp*temp/2);
    }

    /**
     * Compute the updated mean and standard deviation using a fixed-count approximation.
     * @param newPoint The new data to update the distribution 
     * @param N The fixed number of data points in the distribution 
     */
    private void UpdateMeanAndVariance(float newPoint, int N_NOTUSED)
    {
        // Useful constants
        final float Np1Inv = 1f/(N + 1f);
        final float distanceFromMean = newPoint - mean;

        // Compute the updated variance (standard deviation)
        standardDeviation = standardDeviation * standardDeviation + distanceFromMean*distanceFromMean*Np1Inv;
        standardDeviation = (float) Math.sqrt(N*Np1Inv * (standardDeviation));

        // Compute the new mean value 
        mean = (N*mean + newPoint)*Np1Inv;   
        N++;     
    }

    @Override
    protected boolean ShouldSend(float inputSignal) {
        // Use the normalized normal distribution as a measure of how likely  
        return ComputeNormalizedDist(inputSignal) >= rand.nextFloat();
    }

    @Override
    protected float GetOutputStrength() {
        return strength;
    }

    @Override
    protected void UpdateDistribution(float backpropSignal, int N) {
        // Update the distribution mean and variance
        UpdateMeanAndVariance(backpropSignal, N);

    }
    
}
