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
        return mergedForwardStrength;
    }

    /**
     * @return Checks if this node is active and returns the value if it is,
     *         otherwise returns null
     */
    public Double getValueOrNull() {
        return hasValidForwardSignal() ? mergedForwardStrength : null;
    }

    @Override
    public void recieveBackwardSignal(Signal signal) {
        super.recieveBackwardSignal(signal);
    }

    @Override
    public boolean addIncomingConnection(Arc connection)
    {
        boolean b = super.addIncomingConnection(connection);
        
        // set all weights to 1 and all biases to 0
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = 1;
            }
        }

        for (int i = 0; i < biases.length; i++) {
            biases[i] = 0;
        }

        return b;
    }

}
