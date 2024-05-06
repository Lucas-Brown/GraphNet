package com.lucasbrown.GraphNetwork.Local.Nodes;

import java.security.InvalidAlgorithmParameterException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Signal;

public class NodeWrapper implements INode {

    protected INode wrappingNode;

    public NodeWrapper(INode node) {
        wrappingNode = node;
    }

    @Override
    public int getID() {
        return wrappingNode.getID();
    }

    @Override
    public String getName() {
        return wrappingNode.getName();
    }

    @Override
    public void setName(String name) {
        wrappingNode.setName(name);
    }

    @Override
    public void setParentNetwork(GraphNetwork network) {
        wrappingNode.setParentNetwork(network);
    }

    @Override
    public GraphNetwork getParentNetwork() {
        return wrappingNode.getParentNetwork();
    }

    @Override
    public ActivationFunction getActivationFunction() {
        return wrappingNode.getActivationFunction();
    }

    /**
     * 
     * @param node
     * @return whether this node is connected to the provided node
     */
    @Override
    public boolean doesContainConnection(INode node) {
        return wrappingNode.doesContainConnection(node);
    }

    /**
     * Get the arc associated with the transfer from this node to the given
     * recieving node
     * 
     * @param recievingNode
     * @return The arc if present, otherwise null
     */
    @Override
    public Arc getArc(INode recievingNode) {
        return wrappingNode.getArc(recievingNode);
    }

    /**
     * Throws an InvalidAlgorithmParameterException
     * 
     * @param connection
     * @return true
     */
    @Override
    public boolean addIncomingConnection(Arc connection) {
        return wrappingNode.addIncomingConnection(connection);
    }

    @Override
    public Collection<Arc> getAllIncomingConnections() {
        return wrappingNode.getAllIncomingConnections();
    }

    /**
     * Add an outgoing connection to the node
     * 
     * @param connection
     * @return true
     */
    @Override
    public boolean addOutgoingConnection(Arc connection) {
        wrappingNode.addOutgoingConnection(connection);
        return false;
    }

    @Override
    public Collection<Arc> getAllOutgoingConnections() {
        return wrappingNode.getAllOutgoingConnections();
    }

    @Override
    public Optional<Arc> getOutgoingConnectionTo(INode recievingNode) {
        return wrappingNode.getOutgoingConnectionTo(recievingNode);
    }

    /**
     * Notify this node of a new incoming forward signal
     * 
     * @param signal
     */
    @Override
    public void recieveForwardSignal(Signal signal) {
        wrappingNode.recieveForwardSignal(signal);
    }

    /**
     * Notify this node of a new incoming backward signal
     * 
     * @param signal
     */
    @Override
    public void recieveBackwardSignal(Signal signal) {
        wrappingNode.recieveBackwardSignal(signal);
    }

    /**
     * Notify this node of a new inference signal
     * 
     * @param signal
     */
    @Override
    public void recieveInferenceSignal(Signal signal) {
        wrappingNode.recieveInferenceSignal(signal);
    }

    /**
     * Get whether the current forward signal is set and valid
     * 
     * @return
     */
    @Override
    public boolean hasValidForwardSignal() {
        return wrappingNode.hasValidForwardSignal();
    }

    @Override
    public void setValidForwardSignal(boolean state) {
        wrappingNode.setValidForwardSignal(state);
    }

    @Override
    public double[] getWeights(int bitStr) {
        return wrappingNode.getWeights(bitStr);
    }

    @Override
    public double getBias(int bitStr) {
        return wrappingNode.getBias(bitStr);
    }

    @Override
    public void sendForwardSignals() {
        wrappingNode.sendForwardSignals();
    }

    @Override
    public ArrayList<Outcome> getState() {
        return wrappingNode.getState();
    }

    @Override
    public void sendErrorsBackwards(ArrayList<Outcome> outcomesAtTime, int timestep) {
        wrappingNode.sendErrorsBackwards(outcomesAtTime, timestep);
    }

    @Override
    public void applyParameterUpdate() {
        wrappingNode.applyParameterUpdate();
    }

    @Override
    public void applyErrorSignals(double epsilon, List<ArrayList<Outcome>> allOutcomes) {
        wrappingNode.applyErrorSignals(epsilon, allOutcomes);
    }

    @Override
    public void acceptSignals() throws InvalidAlgorithmParameterException {
        wrappingNode.acceptSignals();
    }

    @Override
    public void clearSignals() {
        wrappingNode.clearSignals();
    }

    @Override
    public void sendTrainingSignals() {
        wrappingNode.sendTrainingSignals();
    }

    @Override
    public int compareTo(INode o) {
        return INode.CompareNodes(this, o);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof INode))
            return false;
        return INode.areNodesEqual(this, (INode) o);
    }


}
