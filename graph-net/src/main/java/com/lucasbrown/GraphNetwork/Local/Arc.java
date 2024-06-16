package com.lucasbrown.GraphNetwork.Local;

import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.NetworkTraining.DataSetTraining.IExpectationAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.IFilter;

/**
 * A one-way connection between a sending node and a recieving node.
 * Holds the probability distribution for determining activation likelyhood
 */
public class Arc {

    /**
     * Sending and recieving node
     */
    public final INode sending, recieving;

    /**
     * INode transfer function for determining probability and strength of signal
     * forwarding
     */
    public IFilter filter;

    public IExpectationAdjuster filterAdjuster;

    public Arc(final INode sending, final INode recieving,
            final IFilter filter, IExpectationAdjuster filterAdjuster) {
        this.sending = sending;
        this.recieving = recieving;
        this.filter = filter;
        this.filterAdjuster = filterAdjuster;
    }

    public int getSendingID() {
        return sending.getID();
    }

    public int getRecievingID() {
        return recieving.getID();
    }

    public boolean doesMatchNodes(INode sendingMatch, INode recievingMatch) {
        return sending.equals(sendingMatch) && recieving.equals(recievingMatch);
    }

    /**
     * Send an inference signal from the sending node to the recieving node
     * 
     * @param strength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    /*
     * Signal sendInferenceSignal(double strength, double probability) {
     * Signal signal = new Signal(sending, recieving, strength, probability);
     * recieving.recieveInferenceSignal(signal);
     * return signal;
     * }
     */

    /**
     * Send an forward signal from the sending node to the recieving node
     * 
     * @param strength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    public Signal sendForwardSignal(Outcome sourceOutcome) {
        Signal signal = new Signal(sending, recieving, sourceOutcome,
                sourceOutcome.probability * filter.getChanceToSend(sourceOutcome.netValue));
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
     * Signal sendBackwardSignal(double strength, double probability) {
     * // sending and recieving have reversed meanings here
     * Signal signal = new Signal(recieving, sending, strength, probability);
     * sending.recieveBackwardSignal(signal);
     * return signal;
     * }
     */

    /**
     * Randomly selects whether the signalStrength passes through the filter
     * 
     * @param signalStrength
     * @return
     */
    public boolean rollFilter(double signalStrength) {
        return filter.shouldSend(signalStrength);
    }

}
