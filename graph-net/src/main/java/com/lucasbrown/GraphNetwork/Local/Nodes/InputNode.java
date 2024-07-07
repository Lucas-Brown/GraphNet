package com.lucasbrown.GraphNetwork.Local.Nodes;

import java.util.ArrayList;

import com.lucasbrown.GraphNetwork.Local.Edge;
import com.lucasbrown.GraphNetwork.Local.Outcome;

/**
 * A node which exposes the functionality of recieving signals.
 * InputNodes cannot have any incoming connections
 */
public class InputNode extends NodeWrapper implements IInputNode {

    private double inputValue;
    private ArrayList<Outcome> outcomes;

    public InputNode(ITrainable node) {
        super(node);
    }

    @Override
    public void acceptUserForwardSignal(double value) {
        inputValue = value;
        wrappingNode.getParentNetwork().notifyNodeActivation(this);
        wrappingNode.setValidForwardSignal(true);
        outcomes = new ArrayList<>(1);
        outcomes.add(getOutcome());
    }

    @Override
    public void acceptSignals() {
        // do nothing!
    }

    @Override
    public void sendForwardSignals() {
        for (Edge connection : getAllOutgoingConnections()) {
            connection.sendForwardSignal(outcomes.get(0));
        }
    }

    @Override
    public ArrayList<Outcome> getState() {
        return outcomes;
    }

    private Outcome getOutcome() {
        Outcome outcome = new Outcome();
        outcome.node = this;
        outcome.netValue = inputValue;
        outcome.activatedValue = getActivationFunction().activator(inputValue);
        outcome.binary_string = -1;
        outcome.probability = 1;
        return outcome;
    }

    @Override
    public boolean addIncomingConnection(Edge connection) {
        throw new UnsupportedOperationException("Input nodes are not allowed to have any incoming connections.");
    }

    @Override
    public String toString() {
        return String.format("%s: (%.2e, 100%s)", getName(), inputValue, "%");
    }

    @Override
    public void applyDelta(double[] gradient)
    {
        // do not
    }

}
