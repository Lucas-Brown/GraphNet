package src.GraphNetwork.Local;

/**
 * Contains the probability distribution information for likelyhood of a signal being sent from one node to another
 */
public abstract class ActivationProbabilityDistribution {
    
    /**
     * The output signal strength
     */
    protected double strength;

    public abstract boolean shouldSend(double inputSignal, double factor);
    public abstract double getOutputStrength();
    protected abstract void updateDistribution(double backpropSignal, int N_Limiter);
    public abstract double getMostLikelyValue();

    /**
     * Adjusts the signal strength towards a target value 
     */
    public void adjustSignalStrength(double target, double epsilon)
    {
        strength += (target - strength) * epsilon;
    }

}
