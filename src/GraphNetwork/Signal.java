package src.GraphNetwork;

public final class Signal {
    public final ActivationProbabilityDistribution recievingFunction; 
    public final float strength;
    
    public Signal(final ActivationProbabilityDistribution recievingFunction, final float strength)
    {
        this.recievingFunction = recievingFunction;
        this.strength = strength;
    }

    public double GetOutputStrength()
    {
        return (double) strength;
    }
}
