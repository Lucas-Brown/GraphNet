package com.lucasbrown.GraphNetwork.Local;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.SharedNetworkData;

/**
 * A node which can be given a value to propagate as a signal
 * InputNodes cannot have any incoming connections
 */
public class InputNode extends Node {

    public InputNode(final GraphNetwork network, final SharedNetworkData networkData,
            final ActivationFunction activationFunction) {
        super(network, networkData, activationFunction);
    }

    /**
     * Send a signal to this node with a
     * 
     * @param value
     */
    public void recieveInputSignal(double value) {
        // recieving a signal from null indicates a user-input
        network.createSignal(null, this, value);
        outputStrength = value;
    }

    @Override
    public void acceptIncomingSignals() {
    }

    @Override
    public boolean addIncomingConnection(Arc connection) {
        throw new UnsupportedOperationException("Input nodes are not allowed to have any incoming connections.");
    }

}
