package com.lucasbrown.GraphNetwork.Local.ReferenceStructure;

import java.util.ArrayList;

import com.lucasbrown.GraphNetwork.Global.ReferenceGraphNetwork;
import com.lucasbrown.GraphNetwork.Global.SharedNetworkData;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.IInputNode;
import com.lucasbrown.GraphNetwork.Local.Signal;

/**
 * A node which exposes the functionality of recieving signals.
 * InputNodes cannot have any incoming connections
 */
public class InputReferenceNode extends ReferenceNode implements IInputNode{

    public InputReferenceNode(final ReferenceGraphNetwork network, final SharedNetworkData networkData,
            final ActivationFunction activationFunction) {
        super(network, networkData, activationFunction);
    }

    @Override
    public void acceptUserInferenceSignal(double value) {
        super.recieveInferenceSignal(new Signal(null, this, value));
    }

    @Override
    public void acceptUserForwardSignal(double value) {
        super.recieveForwardSignal(new Signal(null, this, value));
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

}
