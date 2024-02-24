package com.lucasbrown.GraphNetwork.Local;

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
    public final ActivationProbabilityDistribution probDist;

    public Arc(final Node sending, final Node recieving,
            final ActivationProbabilityDistribution transferFunc) {
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
     * @param outputStrength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    Signal sendInferenceSignal(double signalStrength, double outputStrength) {
        Signal signal = new Signal(sending, recieving, outputStrength);
        recieving.recieveInferenceSignal(signal);
        return signal;
    }


    /**
     * Send an forward signal from the sending node to the recieving node
     * 
     * @param outputStrength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    Signal sendForwardSignal(double signalStrength, double outputStrength) {
        Signal signal = new Signal(sending, recieving, outputStrength);
        recieving.recieveForwardSignal(signal);
        return signal;
    }

    /**
     * Send an backward signal from the recieving node to the sending node
     * 
     * @param outputStrength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    Signal sendBackwardSignal(double signalStrength, double outputStrength) {
        // sending and recieving have reversed meanings here
        Signal signal = new Signal(recieving, sending, outputStrength);
        sending.recieveBackwardSignal(signal);
        return signal;
    }

    
    /**
     * Randomly selects whether the signalStrength passes through the filter
     * @param signalStrength
     * @return 
     */
    public boolean rollFilter(double signalStrength) {
        return probDist.shouldSend(signalStrength);
    }


}
