package src.GraphNetwork.Local;

import src.GraphNetwork.Global.GraphNetwork;
import src.GraphNetwork.Global.SharedNetworkData;
import src.NetworkTraining.NodeErrorHandling;

/**
 * A node which exposes it's value to the user and can be sent a correction signal 
 */
public class OutputNode extends Node {

    public OutputNode(final GraphNetwork network, final SharedNetworkData networkData, final ActivationFunction activationFunction) {
        super(network, networkData, activationFunction);
    }

    /**
     * Get whether this node is active (i.e. has a valid value)
     * @return
     */
    public boolean isActive()
    {
        return !incomingSignals.isEmpty();
    }

    /**
     * Get the value of this node
     * The caller should first verify if this node is active using {@code isActive} or get the value using {@code getValueOrNull}
     * @return
     */
    public double getValue()
    {
        return mergedSignalStrength;
    }

    /**
     * @return Checks if this node is active and returns the value if it is, otherwise returns null
     */
    public Double getValueOrNull()
    {
        return isActive() ? mergedSignalStrength : null;
    }

    /**
     * Correct the value of this node
     * Pass a null-value to indicate that this node should NOT have a value currently 
     * @param value the value this node should have 
     */
    public void correctOutputValue(Double target)
    {
        if(isActive())
        {
            if(target == null)
            {
                NodeErrorHandling.diminishFiringChances(history, this);
            }
            else
            {
                NodeErrorHandling.correctSignalValue(target);
            }
        }
        else
        {
            NodeErrorHandling.sendErrorSignal();
        }

        network.createSignal(null, this, target); 
        super.mergedSignalStrength = target;
    }

    
}
