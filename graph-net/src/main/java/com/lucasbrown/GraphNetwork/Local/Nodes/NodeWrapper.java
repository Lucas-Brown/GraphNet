package com.lucasbrown.GraphNetwork.Local.Nodes;

import java.security.InvalidAlgorithmParameterException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Optional;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Edge;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Signal;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.SignalCombinator;

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
     * Throws an InvalidAlgorithmParameterException
     * 
     * @param connection
     * @return true
     */
    @Override
    public boolean addIncomingConnection(Edge connection) {
        return wrappingNode.addIncomingConnection(connection);
    }

    @Override
    public Collection<Edge> getAllIncomingConnections() {
        return wrappingNode.getAllIncomingConnections();
    }

    /**
     * Add an outgoing connection to the node
     * 
     * @param connection
     * @return true
     */
    @Override
    public boolean addOutgoingConnection(Edge connection) {
        wrappingNode.addOutgoingConnection(connection);
        return false;
    }

    @Override
    public Collection<Edge> getAllOutgoingConnections() {
        return wrappingNode.getAllOutgoingConnections();
    }

    @Override
    public Optional<Edge> getOutgoingConnectionTo(INode recievingNode) {
        return wrappingNode.getOutgoingConnectionTo(recievingNode);
    }

    @Override
    public Optional<Edge> getIncomingConnectionFrom(INode sendingNode) {
        return wrappingNode.getIncomingConnectionFrom(sendingNode);
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
    public void sendForwardSignals() {
        wrappingNode.sendForwardSignals();
    }

    @Override
    public ArrayList<Outcome> getState() {
        return wrappingNode.getState();
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
    public int compareTo(INode o) {
        return INode.CompareNodes(this, o);
    }

    @Override
    public int hashCode(){
        return wrappingNode.hashCode();
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof INode))
            return false;
        return INode.areNodesEqual(this, (INode) o);
    }

    @Override
    public int nodeSetToBinStr(Collection<INode> incomingNodes) {
        return wrappingNode.nodeSetToBinStr(incomingNodes);
    }

    @Override
    public ArrayList<Edge> binStrToArcList(int binStr) {
        return wrappingNode.binStrToArcList(binStr);
    }

    @Override
    public SignalCombinator getCombinator(){
        return wrappingNode.getCombinator();
    }

}
