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

    private double N;
    private double N_Limiter;

    /**
     * @param mean mean value of the normal distribution
     * @param standardDeviation standard deviation of the normal distribution
     */
    public BellCurveDistribution(double mean, double standardDeviation, double N_Limiter)
    {
        this.mean = mean;
        this.standardDeviation = standardDeviation;
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
        final double temp = (x-mean)/standardDeviation;
        return (double) Math.exp(-temp*temp/2);
    }

    /**
     * Reinforce the mean and standard deviation with {@code valueToReinforce} using a fixed-count approximation.
     * @param valueToReinforce The new data to 'add' to the distribution data set
     * @param N_Limiter The fixed number of data points in the distribution 
     */
    @Override
    public void reinforceDistribution(double valueToReinforce, double reinforcmentRate)
    {
        // Useful constants
        final double Np1Inv = 1.0/(N + reinforcmentRate);
        final double distanceFromMean = valueToReinforce - mean;

        // Compute the updated variance (standard deviation)
        standardDeviation = standardDeviation * standardDeviation + distanceFromMean*distanceFromMean*Np1Inv;
        standardDeviation = (double) Math.sqrt(N*Np1Inv * (standardDeviation));

        // Compute the new mean value 
        mean = (N*mean + valueToReinforce)*Np1Inv;   

        N += reinforcmentRate;
        if(N > N_Limiter)
        {
            N = N_Limiter;
        } 
    }

    /**
     * Diminish the mean and standard deviation with {@code valueToDiminish} using a fixed-count approximation.
     * @param valueToDiminish The new data to 'remove' from the distribution data set
     * @param N The fixed number of data points in the distribution 
     */
    @Override
    public void diminishDistribution(double valueToDiminish, double diminishmentRate)
    {
        // Useful constants
        final double Nm1Inv = 1.0/(N - diminishmentRate);
        final double distanceFromMean = valueToDiminish - mean;

        // Compute the updated variance (standard deviation)
        standardDeviation = standardDeviation * standardDeviation - distanceFromMean*distanceFromMean/(N*N);
        standardDeviation = (double) Math.sqrt(N*Nm1Inv * (standardDeviation));

        // Compute the new mean value 
        mean = (N*mean - valueToDiminish)*Nm1Inv;   

        N -= diminishmentRate;
        if(N < 1)
        {
            N = 1;
        } 
    }

    @Override
    public boolean shouldSend(double inputSignal, double factor) {
        // Use the normalized normal distribution as a measure of how likely  
        return factor*computeNormalizedDist(inputSignal) >= rand.nextDouble();
    }

    @Override
    public double getMostLikelyValue()
    {
        return mean;
    }
    
}
