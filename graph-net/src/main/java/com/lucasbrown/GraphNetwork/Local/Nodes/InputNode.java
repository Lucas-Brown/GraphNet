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

    public InputNode(final GraphNetwork network, final ActivationFunction activationFunction) {
        super(network, activationFunction);
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
            connection.sendForwardSignal(-1, activationFunction.activator(inputValue), connection.probDist.sendChance(inputValue)); 
        }
    }

    @Override
    public ArrayList<Outcome> getState()
    {
        outcomes = new ArrayList<>(1);
        Outcome outcome = new Outcome();
        outcome.netValue = inputValue;
        outcome.activatedValue = activationFunction.activator(inputValue);
        outcome.binary_string = -1;
        outcome.probability = 1;
        outcomes.add(outcome);
        return outcomes;
    }

    @Override 
    public void sendErrorsBackwards(ArrayList<Outcome> outcomesAtTime, int timestep){
        // since this is an input node, there's nothing to send an error to
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

    
    @Override
    public String toString() {
        return name + ": (" + inputValue + ", 100%)";
    }

}
