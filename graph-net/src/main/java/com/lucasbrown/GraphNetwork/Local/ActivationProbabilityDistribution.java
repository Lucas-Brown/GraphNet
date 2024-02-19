package com.lucasbrown.GraphNetwork.Local;

import java.util.Random;

/**
 * Contains the probability distribution information for likelyhood of a signal
 * being sent from one node to another
 */
public abstract class ActivationProbabilityDistribution {

    /**
     * Random number generator for probabalistically choosing whether to send a
     * signal
     */
    protected Random rand = new Random();

    public abstract boolean shouldSend(double inputSignal);

    public abstract void reinforceDistribution(double valueToReinforce);

    public abstract void diminishDistribution(double valueToDiminish);

    public abstract double getMeanValue();

}
