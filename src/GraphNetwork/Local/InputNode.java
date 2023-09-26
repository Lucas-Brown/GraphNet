package src.GraphNetwork.Local;

import src.GraphNetwork.Global.GraphNetwork;
import src.GraphNetwork.Global.SharedNetworkData;

/**
 * An input node accepts user data and generates a forward-propogating history.
 */
public class InputNode extends Node {

    public InputNode(final GraphNetwork network, final SharedNetworkData networkData, final ActivationFunction activationFunction) {
        super(network, networkData, activationFunction);
    }

    public void recieveInputSignal()
    {
        
    }
    
}
