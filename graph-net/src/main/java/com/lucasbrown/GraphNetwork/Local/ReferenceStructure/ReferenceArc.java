package com.lucasbrown.GraphNetwork.Local.ReferenceStructure;

import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.FilterDistribution;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.GraphNetwork.Local.Signal;

/**
 * A one-way connection between a sending node and a recieving node.
 * Holds the probability distribution for determining activation likelyhood
 */
public class ReferenceArc extends Arc {

    /**
     * Sending and recieving node
     */
    final Node sending, recieving;

    public ReferenceArc(final Node sending, final Node recieving,
            final FilterDistribution transferFunc) {
        super(transferFunc);

        this.sending = sending;
        this.recieving = recieving;
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
    @Override
    public Signal sendInferenceSignal(double strength) {
        Signal signal = new Signal(sending, recieving, strength);
        recieving.recieveInferenceSignal(signal);
        return signal;
    }

    /**
     * Send an forward signal from the sending node to the recieving node
     * 
     * @param strength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    @Override
    public Signal sendForwardSignal(double strength) {
        Signal signal = new Signal(sending, recieving, strength);
        recieving.recieveForwardSignal(signal);
        return signal;
    }

    /**
     * Send an backward signal from the recieving node to the sending node
     * 
     * @param strength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    @Override
    public Signal sendBackwardSignal(double strength) {
        // sending and recieving have reversed meanings here
        Signal signal = new Signal(recieving, sending, strength);
        sending.recieveBackwardSignal(signal);
        return signal;
    }

    @Override
    public Node getSendingNode() {
        return sending;
    }

    @Override
    public Node getRecievingNode() {
        return recieving;
    }

    @Override
    public int getSendingID() {
        return sending.id;
    }

    @Override
    public int getRecievingID() {
        return recieving.id;
    }

}
