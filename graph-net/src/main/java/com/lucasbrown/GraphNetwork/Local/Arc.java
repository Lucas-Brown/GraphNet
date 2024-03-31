package com.lucasbrown.GraphNetwork.Local;

import com.lucasbrown.GraphNetwork.Distributions.FilterDistribution;

/**
 * A one-way connection between a sending node and a recieving node.
 * Holds the probability distribution for determining activation likelyhood
 */
public class Arc {

    /**
     * Sending and recieving node
     */
    final Node sending, recieving;

    /**
     * Node transfer function for determining probability and strength of signal
     * forwarding
     */
    public final FilterDistribution probDist;

    public Arc(final Node sending, final Node recieving,
            final FilterDistribution transferFunc) {
        this.sending = sending;
        this.recieving = recieving;
        this.probDist = transferFunc;
    }

    public boolean doesMatchNodes(Node sendingMatch, Node recievingMatch) {
        return sending.equals(sendingMatch) && recieving.equals(recievingMatch);
    }

    /**
     * Send an inference signal from the sending node to the recieving node
     * 
     * @param strength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    /* 
    Signal sendInferenceSignal(double strength, double probability) {
        Signal signal = new Signal(sending, recieving, strength, probability);
        recieving.recieveInferenceSignal(signal);
        return signal;
    }
    */


    /**
     * Send an forward signal from the sending node to the recieving node
     * 
     * @param strength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    Signal sendForwardSignal(int sourceKey, double strength, double probability) {
        Signal signal = new Signal(sending, recieving, sourceKey, strength, probability);
        recieving.recieveForwardSignal(signal);
        return signal;
    }

    /**
     * Send an backward signal from the recieving node to the sending node
     * 
     * @param strength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    /* 
    Signal sendBackwardSignal(double strength, double probability) {
        // sending and recieving have reversed meanings here
        Signal signal = new Signal(recieving, sending, strength, probability);
        sending.recieveBackwardSignal(signal);
        return signal;
    }
    */

    
    /**
     * Randomly selects whether the signalStrength passes through the filter
     * @param signalStrength
     * @return 
     */
    public boolean rollFilter(double signalStrength) {
        return probDist.shouldSend(signalStrength);
    }


}
