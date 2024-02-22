package com.lucasbrown.GraphNetwork.Local;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.SharedNetworkData;

/**
 * A node which exposes it's value and can be sent a corrective (backward)
 * signal
 */
public class OutputNode extends Node {

    public OutputNode(final GraphNetwork network, final SharedNetworkData networkData,
            final ActivationFunction activationFunction) {
        super(network, networkData, activationFunction);
    }

    /**
     * Get the value of this node
     * The caller should first verify if this node is active using {@code isActive}
     * or get the value using {@code getValueOrNull}
     * 
     * @return
     */
    public double getValue() {
        return outputStrength;
    }

    /**
     * @return Checks if this node is active and returns the value if it is,
     *         otherwise returns null
     */
    public Double getValueOrNull() {
        return hasValidForwardSignal() ? outputStrength : null;
    }

    @Override
    void recieveBackwardSignal(Signal signal) {
        super.recieveBackwardSignal(signal);
    }

}
