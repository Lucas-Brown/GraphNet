package com.lucasbrown.GraphNetwork.Local;

/**
 * A one-way connection between a sending node and a recieving node.
 * Holds the probability distribution for determining activation likelyhood
 */
public abstract class Arc {

    /**
     * Node transfer function for determining probability and strength of signal
     * forwarding
     */
    public final FilterDistribution probDist;

    public Arc(final FilterDistribution transferFunc) {
        this.probDist = transferFunc;
    }

    /**
     * Send an inference signal from the sending node to the recieving node
     * 
     * @param strength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    public abstract Signal sendInferenceSignal(double strength);


    /**
     * Send an forward signal from the sending node to the recieving node
     * 
     * @param strength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    public abstract Signal sendForwardSignal(double strength);

    /**
     * Send an backward signal from the recieving node to the sending node
     * 
     * @param strength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    public abstract Signal sendBackwardSignal(double strength);

    public abstract Node getSendingNode();

    public abstract Node getRecievingNode();
    
    public abstract int getSendingID();

    public abstract int getRecievingID();
    
    /**
     * Randomly selects whether the signalStrength passes through the filter
     * @param signalStrength
     * @return 
     */
    public boolean rollFilter(double signalStrength) {
        return probDist.shouldSend(signalStrength);
    }


}
