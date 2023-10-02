package src.GraphNetwork.Local;

/**
 * Contains the probability distribution information for likelyhood of a signal being sent from one node to another
 */
public abstract class ActivationProbabilityDistribution {
    
    public abstract boolean shouldSend(double inputSignal, double factor);
    public abstract void reinforceDistribution(double valueToReinforce, double reinforcmentRate);
    public abstract void diminishDistribution(double valueToDiminish, double diminishmentRate);
    public abstract double getMostLikelyValue();

}
