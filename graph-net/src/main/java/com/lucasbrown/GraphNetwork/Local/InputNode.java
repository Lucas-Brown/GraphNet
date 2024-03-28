package com.lucasbrown.GraphNetwork.Local;

import java.security.InvalidAlgorithmParameterException;
import java.util.ArrayList;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.SharedNetworkData;

/**
 * A node which exposes the functionality of recieving signals.
 * InputNodes cannot have any incoming connections
 */
public class InputNode extends Node implements IInputNode {

    private double inputValue;

    public InputNode(final GraphNetwork network, final SharedNetworkData networkData,
            final ActivationFunction activationFunction) {
        super(network, networkData, activationFunction);
    }

    @Override
    public void acceptUserForwardSignal(double value) {
        inputValue = value;
        network.notifyNodeActivation(this);
        hasValidForwardSignal = true;
    }

    @Override
    public void acceptUserInferenceSignal(double value) {
        inputValue = value;
        network.notifyNodeActivation(this);
        hasValidForwardSignal = true;
    }

    @Override
    public void acceptSignals() {
        // do nothing!
    }

    @Override
    public void sendForwardSignals() {
        for (Arc connection : outgoing) {
            connection.sendForwardSignal(activationFunction.activator(inputValue), connection.probDist.sendChance(inputValue)); 
        }
    }
    /*
     * @Override
     * protected void acceptIncomingForwardSignals(ArrayList<Signal>
     * incomingSignals) {
     * if (incomingSignals.size() == 0)
     * return;
     * 
     * assert incomingSignals.size() == 1;
     * 
     * super.hasValidForwardSignal = true;
     * 
     * mergedForwardStrength = incomingSignals.get(0).strength;
     * activatedStrength = activationFunction.activator(mergedForwardStrength);
     * }
     * 
     * @Override
     * protected void updateWeightsAndBias(double error_derivative){
     * return; // do not update, no matter what *HE* whispers
     * }
     */

    @Override
    public boolean addIncomingConnection(Arc connection) {
        throw new UnsupportedOperationException("Input nodes are not allowed to have any incoming connections.");
    }

}
