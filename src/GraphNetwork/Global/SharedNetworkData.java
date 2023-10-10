package src.GraphNetwork.Global;

import src.NetworkTraining.ErrorFunction;

/**
 * A collection of data which modifies the training and firing rates of each node.
 * All nodes are given access to this data to use but modification should be strictly controlled by {@code GraphNetwork}
 */
public class SharedNetworkData
{

    public final ErrorFunction errorFunc;

    /**
     * The weight of adding new data points to the distribution 
     */
    private double reinforcmentRate;

    /**
     * Step size for adjusting the output values of nodes
     */
    private double epsilon;

    /**
     * The factor of decay for the likelyhood of a node firing sucessive signals in one step
     * i.e. The first check is unchanged, the second check is multiplied by a factor of likelyhoodDecay, the third a factor of likelyhoodDecay * likelyhoodDecay and so on.
     */
    private double likelyhoodDecay;

    /**
     * Dynamically adjusts the firing rate of the network  
     */
    private double globalFiringRateMultiplier;

    SharedNetworkData(ErrorFunction errorFunc, double reinforcmentRate, double epsilon, double likelyhoodDecay, double globalFiringRateMultiplier)
    {
        this.errorFunc = errorFunc;
        this.reinforcmentRate = reinforcmentRate;
        this.epsilon = epsilon;
        this.likelyhoodDecay = likelyhoodDecay;
        this.globalFiringRateMultiplier = globalFiringRateMultiplier;
    }

    public double getReinforcmentRate()
    {
        return reinforcmentRate;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public double getLikelyhoodDecay() {
        return likelyhoodDecay;
    }

    public double getGlobalFiringRateMultiplier() {
        return globalFiringRateMultiplier;
    }
}