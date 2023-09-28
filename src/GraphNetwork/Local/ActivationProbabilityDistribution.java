package src.GraphNetwork.Local;

/**
 * Contains the probability distribution information for likelyhood of a signal being sent from one node to another
 */
public abstract class ActivationProbabilityDistribution {
    
    public abstract boolean shouldSend(double inputSignal, double factor);
    public abstract void reinforceDistribution(double valueToReinforce, int N_Limiter);
    public abstract void diminishDistribution(double valueToDiminish);
    public abstract double getMostLikelyValue();

}
