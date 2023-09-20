package src.GraphNetwork.Local;

/**
 * Contains the probability distribution information for likelyhood of a signal being sent from one node to another
 */
public abstract class ActivationProbabilityDistribution {
    
    /**
     * The output signal strength
     */
    protected float strength;

    public abstract boolean shouldSend(float inputSignal, float factor);
    public abstract float getOutputStrength();
    protected abstract void updateDistribution(float backpropSignal, int N_Limiter);
    public abstract float getMostLikelyValue();

    /**
     * Adjusts the signal strength towards a target value 
     */
    public void adjustSignalStrength(float target, float epsilon)
    {
        strength += (target - strength) * epsilon;
    }

}
