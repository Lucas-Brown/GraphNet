package com.lucasbrown.GraphNetwork.Local.DataStructure;

import java.util.Arrays;

import com.lucasbrown.GraphNetwork.Global.DataGraphNetwork;
import com.lucasbrown.GraphNetwork.Global.SharedNetworkData;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Signal;

/**
 * A node which exposes it's value and can be sent a corrective (backward)
 * signal
 */
public class OutputDataNode extends DataNode {

    public OutputDataNode(final DataGraphNetwork network, final SharedNetworkData networkData,
            final ActivationFunction activationFunction, int id) {
        super(network, networkData, activationFunction, id);
    }
    
    public OutputDataNode(DataNode toCopy)
    {
        super(toCopy);
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
    public boolean addIncomingConnection(Arc connection) {
        boolean b = super.addIncomingConnection(connection);

        // set all weights to 1 and all biases to 0
        nodeSetToWeightsAndBias.values().forEach(wAb -> {
            wAb.bias = 0;
            Arrays.fill(wAb.weights, 1);
        });

        return b;
    }

    @Override
    public OutputDataNode copy() {
        return new OutputDataNode(this);
    }
}
