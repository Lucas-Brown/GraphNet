package com.lucasbrown.GraphNetwork.Local.DataStructure;

import com.lucasbrown.GraphNetwork.Global.DataGraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.FilterDistribution;
import com.lucasbrown.GraphNetwork.Local.ICopyable;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.GraphNetwork.Local.Signal;

/**
 * A one-way connection between a sending node and a recieving node.
 * Holds the probability distribution for determining activation likelyhood
 */
public class DataArc extends Arc implements ICopyable<DataArc>{

    public DataGraphNetwork graphNetwork;

    /**
     * Sending and recieving node
     */
    final int sending, recieving;

    public DataArc(DataGraphNetwork graphNetwork, final int sending, final int recieving,
            final FilterDistribution transferFunc) {
        super(transferFunc);

        this.graphNetwork = graphNetwork;
        this.sending = sending;
        this.recieving = recieving;
    }

    public DataArc(DataArc toCopy){
        super(toCopy.probDist.copy());
        graphNetwork = toCopy.graphNetwork;
        sending = toCopy.sending;
        recieving = toCopy.recieving;
    }

    /**
     * Send an inference signal from the sending node to the recieving node
     * 
     * @param strength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    @Override
    public Signal sendInferenceSignal(double strength) {
        return graphNetwork.sendInferenceSignalToNode(sending, recieving, strength);
    }


    /**
     * Send an forward signal from the sending node to the recieving node
     * 
     * @param strength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
     @Override
     public Signal sendForwardSignal(double strength) {
        return graphNetwork.sendForwardSignalToNode(sending, recieving, strength);
    }

    /**
     * Send an backward signal from the recieving node to the sending node
     * 
     * @param strength The strength of the signal to send
     * @return the signal or null if no signal was sent
     */
    @Override
     public Signal sendBackwardSignal(double strength) {
        return graphNetwork.sendBackwardSignalToNode(sending, recieving, strength);
    }

    @Override
    public Node getSendingNode() {
        return graphNetwork.getNode(sending);
    }

    @Override
    public Node getRecievingNode() {
        return graphNetwork.getNode(recieving);
    }

    @Override
    public int getSendingID() {
        return sending;
    }

    @Override
    public int getRecievingID() {
        return recieving;
    }

    @Override
    public DataArc copy() {
        return new DataArc(this);
    }

    @Override
    public boolean equals(Object o)
    {
        if(!(o instanceof DataArc)) return false;

        DataArc dArc = (DataArc) o;
        return dArc.recieving == recieving && dArc.sending == sending;
    }

}
