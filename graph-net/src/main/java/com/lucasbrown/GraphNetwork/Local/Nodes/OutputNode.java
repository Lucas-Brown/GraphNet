package com.lucasbrown.GraphNetwork.Local.Nodes;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.SharedNetworkData;
import com.lucasbrown.GraphNetwork.Local.Signal;

/**
 * A node which exposes it's value and can be sent a corrective (backward)
 * signal
 */
public class OutputNode extends NodeWrapper implements IOutputNode {

    public OutputNode(INode node) {
        super(node);
    }

    /**
     * Get the value of this node
     * The caller should first verify if this node is active using {@code isActive}
     * or get the value using {@code getValueOrNull}
     * 
     * @return
     */
    @Override
    public double getValue() {
        return 0; // TODO: fix this
    }

    /**
     * @return Checks if this node is active and returns the value if it is,
     *         otherwise returns null
     */
    @Override
    public Double getValueOrNull() {
        return hasValidForwardSignal() ? 0d : null;
    }

    @Override
    public void acceptUserBackwardSignal(double value) {
        super.recieveBackwardSignal(new Signal(this, null, -1, value, 1));
    }

    /*
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
    */
}
