package com.lucasbrown.GraphNetwork.Local;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.SharedNetworkData;

/**
 * A node which exposes the functionality of recieving signals.
 * InputNodes cannot have any incoming connections
 */
public class InputNode extends Node {

    public InputNode(final GraphNetwork network, final SharedNetworkData networkData,
            final ActivationFunction activationFunction) {
        super(network, networkData, activationFunction);
    }

    @Override
    public void recieveForwardSignal(Signal signal) {
        super.recieveForwardSignal(signal);
    }

    @Override
    void recieveInferenceSignal(Signal signal) {
        super.recieveInferenceSignal(signal);
    }

    @Override
    public boolean addIncomingConnection(Arc connection) {
        throw new UnsupportedOperationException("Input nodes are not allowed to have any incoming connections.");
    }

}
