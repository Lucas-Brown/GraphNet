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

    /**
     * @param mean mean value of the normal distribution
     * @param standardDeviation standard deviation of the normal distribution
     */
    public NormalTransferFunction(float mean, float standardDeviation, float strength)
    {
        this.mean = mean;
        this.standardDeviation = standardDeviation;
        this.strength = strength;
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
     * @param backpropSignal The new data to update the distribution 
     * @param N The fixed number of data points in the distribution 
     */
    private void UpdateMeanAndVariance(float backpropSignal, int N)
    {
        // Useful constant
        final float Np1 = N + 1f;

        // Compute the new mean value and store the old mean for later
        final float oldMean = mean;
        mean = (N*mean + backpropSignal)/Np1;

        // Compute the updated variance (standard deviation)
        float varSqr = N/Np1 * standardDeviation * standardDeviation;
        varSqr += 2*(mean - oldMean)/(backpropSignal - oldMean);
        varSqr += (N+2f)/Np1;

        standardDeviation = (float) Math.sqrt(varSqr);
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
