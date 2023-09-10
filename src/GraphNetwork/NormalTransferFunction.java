package src.GraphNetwork;

import java.util.Random;

public class NormalTransferFunction implements NodeTransferFunction {

    /**
     * Random number generator for probabalistically choosing whether to send a signal
     */
    private Random rand = new Random();

    /**
     * mean value and standard deviation of a normal distribution
     */
    private float mean, standardDeviation;

    /**
     * The output signal 
     */
    private float signal;

    /**
     * @param mean mean value of the normal distribution
     * @param standardDeviation standard deviation of the normal distribution
     */
    public NormalTransferFunction(float mean, float standardDeviation, float signal)
    {
        this.mean = mean;
        this.standardDeviation = standardDeviation;
        this.signal = signal;
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

    @Override
    public boolean ShouldSend(float inputSignal) {
        // Use the normalized normal distribution as a measure of how likely  
        return ComputeNormalizedDist(inputSignal) >= rand.nextFloat();
    }

    @Override
    public float GetOutputSignal() {
        return signal;
    }
    
}
