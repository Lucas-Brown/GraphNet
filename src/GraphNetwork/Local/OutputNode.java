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
     * Get the value of this node
     * The caller should first verify if this node is active using {@code isActive} or get the value using {@code getValueOrNull}
     * @return
     */
    public double getValue()
    {
        return outputStrength;
    }

    /**
     * @return Checks if this node is active and returns the value if it is, otherwise returns null
     */
    public Double getValueOrNull()
    {
        return isActive() ? outputStrength : null;
    }

    /**
     * Correct the value of this node
     * Pass a null-value to indicate that this node should NOT have a value currently 
     * @param value the value this node should have 
     */
    public void correctOutputValue(Double target)
    {
        if(isActive() && history != null)
        {
            if(target == null)
            {
                NodeErrorHandling.diminishFiringChances(history, this);
            }
            else
            {
                NodeErrorHandling.computeErrorSignalsOfHistory(history, this, target);
                NodeErrorHandling.reinforceFiringChances(history, this);
            }
            history.decimateTimeline(this); // remove this timeline
        }
        else
        {
            //NodeErrorHandling.sendErrorSignal();
        }

        //outputStrength = target; // TODO: either enforce no outgoing connections from output nodes or set strength BEFORE signals get sent
    }

    
}
