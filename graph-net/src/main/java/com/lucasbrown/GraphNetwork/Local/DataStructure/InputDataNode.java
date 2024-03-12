package com.lucasbrown.GraphNetwork.Local.DataStructure;

import java.util.ArrayList;

import com.lucasbrown.GraphNetwork.Global.DataGraphNetwork;
import com.lucasbrown.GraphNetwork.Global.SharedNetworkData;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Signal;

/**
 * A node which exposes the functionality of recieving signals.
 * InputNodes cannot have any incoming connections
 */
public class InputDataNode extends DataNode{

    public InputDataNode(final DataGraphNetwork network, final SharedNetworkData networkData,
            final ActivationFunction activationFunction, int id) {
        super(network, networkData, activationFunction, id);
    }

    public InputDataNode(DataNode toCopy)
    {
        super(toCopy);
    }

    @Override
    public void recieveForwardSignal(Signal signal) {
        super.recieveForwardSignal(signal);
    }

    @Override
    public void recieveInferenceSignal(Signal signal) {
        super.recieveInferenceSignal(signal);
    }

    @Override
    protected void acceptIncomingForwardSignals(ArrayList<Signal> incomingSignals) {
        if (incomingSignals.size() == 0)
            return;

        assert incomingSignals.size() == 1;

        super.hasValidForwardSignal = true;

        mergedForwardStrength = incomingSignals.get(0).strength;
        outputStrength = activationFunction.activator(mergedForwardStrength);
    }
    
    @Override
    protected void updateWeightsAndBias(double error_derivative){
        return; // do not update, no matter what *HE* whispers
    }

    @Override
    public boolean addIncomingConnection(Arc connection) {
        throw new UnsupportedOperationException("Input nodes are not allowed to have any incoming connections.");
    }

    @Override
    public InputDataNode copy() {
        return new InputDataNode(this);
    }

}
