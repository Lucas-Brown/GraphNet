package com.lucasbrown.GraphNetwork.Local;

import com.lucasbrown.GraphNetwork.Local.Nodes.INode;

/**
 * A one-way connection between a sending node and a recieving node.
 * Holds the probability distribution for determining activation likelyhood
 */
public class Edge {

    /**
     * Sending and recieving node
     */
    public final INode sending, recieving;

    public Edge(final INode sending, final INode recieving) {
        this.sending = sending;
        this.recieving = recieving;
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
     * Send an forward signal from the sending node to the recieving node
     * 
     * @param strength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    public Signal sendForwardSignal(Outcome sourceOutcome) {
        Signal signal = new Signal(sending, recieving, sourceOutcome);
        recieving.recieveForwardSignal(signal);
        return signal;
    }


}
