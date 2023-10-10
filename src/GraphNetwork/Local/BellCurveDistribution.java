package src.GraphNetwork.Local;

import java.util.Random;

/**
 * A bell curve distribution 
 */
public class BellCurveDistribution extends ActivationProbabilityDistribution {

    /**
     * mean value and standard deviation of a normal distribution
     */
    private double mean, variance;

    private double N;
    private double N_Limiter;

    /**
     * @param mean mean value of the normal distribution
     * @param variance variance of the normal distribution
     */
    public BellCurveDistribution(double mean, double variance, double N_Limiter)
    {
        this.mean = mean;
        this.variance = variance;
        this.N_Limiter = N_Limiter;
        N = 100;
    }

    /**
     * normalized normal distribution
     * @param x 
     * @return value of the distribution at the point x. always returns on the interval (0, 1]
     */
    private double computeNormalizedDist(double x)
    {
        //final double coef = 1/(standardDeviation*Math.sqrt(2*Math.PI));
        final double d = x-mean;
        return Math.exp(-d*d/variance/2);
    }

    /**
     * Reinforce the mean and standard deviation with {@code valueToReinforce}.
     * @param valueToReinforce The new data to add to the distribution data set
     */
    @Override
    public void reinforceDistribution(double valueToReinforce)
    {   
        // Useful constants
        final double weight = 1; // the weight of each new data point.
        final double NpDelInv = 1.0/(N + weight); 
        final double distanceFromMean = valueToReinforce - mean;
        final double shift = weight * distanceFromMean * NpDelInv;

        // Compute the updated variance (standard deviation)
        variance += shift*shift*(N+1);
        variance *= N*NpDelInv;

        // Compute the new mean value 
        mean += shift;   

        N += weight;
        if(N > N_Limiter)
        {
            N = N_Limiter;
        } 
    }
    
    @Override
    public void diminishDistribution(double valueToDiminish) 
    {
        // do not make a change? 
        // reinforcing this distribution with a weight of -1 would effectively act as removing a data point from the distribution
        // unfortunately, doing this just shifts the distribution away from the reinforced value, causing failiure to converge 
    }

    @Override
    public boolean shouldSend(double inputSignal, double factor) {
        // Use the normalized normal distribution as a measure of how likely  
        return factor*computeNormalizedDist(inputSignal) >= rand.nextDouble();
    }

    @Override
    public double getDirectionOfDecreasingLikelyhood(double x)
    {
        return x > mean ? 1 : -1;
    }


}
