package com.lucasbrown.GraphNetwork.Local.Nodes;

import java.security.InvalidAlgorithmParameterException;
import java.util.ArrayList;
import java.util.List;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.SharedNetworkData;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Outcome;

/**
 * A node which exposes the functionality of recieving signals.
 * InputNodes cannot have any incoming connections
 */
public class InputNode extends NodeWrapper implements IInputNode {

    private double inputValue;

    public InputNode(INode node){
        super(node);
    }

    @Override
    public void acceptUserForwardSignal(double value) {
        inputValue = value;
        wrappingNode.getParentNetwork().notifyNodeActivation(this);
        wrappingNode.setValidForwardSignal(true); 
    }

    @Override
    public void acceptUserInferenceSignal(double value) {
        inputValue = value;
        wrappingNode.getParentNetwork().notifyNodeActivation(this);
        wrappingNode.setValidForwardSignal(true); 
    }

    @Override
    public void acceptSignals() {
        // do nothing!
    }

    @Override 
    public void sendTrainingSignals(){
        sendForwardSignals();
        setValidForwardSignal(false);
    }

    @Override
    public void sendForwardSignals() {
        for (Arc connection : getAllOutgoingConnections()) {
            connection.sendForwardSignal(getOutcome()); 
        }
    }

    @Override
    public ArrayList<Outcome> getState()
    {
        ArrayList<Outcome> outcomes = new ArrayList<>(1);
        outcomes.add(getOutcome());
        return outcomes;
    }

    
    private Outcome getOutcome(){
        Outcome outcome = new Outcome();
        outcome.netValue = inputValue;
        outcome.activatedValue = getActivationFunction().activator(inputValue);
        outcome.binary_string = -1;
        outcome.probability = 1;
        return outcome;
    }

    @Override 
    public void sendErrorsBackwards(ArrayList<Outcome> outcomesAtTime, int timestep){
        // since this is an input node, there's nothing to send an error to
    }
    
    @Override
    public void applyErrorSignals(double epsilon, List<ArrayList<Outcome>> allOutcomes) {
        // cannot apply any error
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
        return String.format("%s: (%.2e, 100%s)", getName(), inputValue, "%");
    }

    
}
