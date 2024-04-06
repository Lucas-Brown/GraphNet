package com.lucasbrown.GraphNetwork.Local.Nodes;

import com.lucasbrown.GraphNetwork.Local.Nodes.INode;

public class NodeWrapper implements INode {
    
    private INode wrappingNode;

    public NodeWrapper(INode node){
        wrappingNode = node;
    }
    
    public int getID(){
        return wrappingNode.getID();
    }

    public void setName(String name){
        wrappingNode.setName(name);
    }

    /**
     * 
     * @param node
     * @return whether this node is connected to the provided node
     */
    public boolean doesContainConnection(INode node){
        return wrappingNode.doesContainConnection(node);
    }

    /**
     * Get the arc associated with the transfer from this node to the given
     * recieving node
     * 
     * @param recievingNode
     * @return The arc if present, otherwise null
     */
    public Arc getArc(INode recievingNode){
        return wrappingNode.getArc(recievingNode);
    }

    /**
     * Throws an InvalidAlgorithmParameterException
     * 
     * @param connection
     * @return true
     */
    public boolean addIncomingConnection(Arc connection){
        throw InvalidAlgorithmParameterException("Input nodes cannot have an incoming connection");
    }

    /**
     * Add an outgoing connection to the node
     * 
     * @param connection
     * @return true
     */
    public boolean addOutgoingConnection(Arc connection){
        wrappingNode.addOutgoingConnection(connection);
    }

    public Optional<Arc> getOutgoingConnectionTo(Node recievingNode){
        return wrappingNode.getOutgoingConnectionTo(recievingNode);
    }

    /**
     * Notify this node of a new incoming forward signal
     * 
     * @param signal
     */
    public void recieveForwardSignal(Signal signal){
        wrappingNode.recieveForwardSignal(signal);
    }

    public void recieveError(int timestep, int key, double error){
        wrappingNode.recieveError(timestep, key, error);
    }

    public Double getError(int timestep, int key){
        return wrappingNode.getError(timestep, key);
    }

    /**
     * Notify this node of a new incoming backward signal
     * 
     * @param signal
     */
    public void recieveBackwardSignal(Signal signal){
        wrappingNode.recieveBackwardSignal(signal);
    }

    /**
     * Notify this node of a new inference signal
     * 
     * @param signal
     */
    public void recieveInferenceSignal(Signal signal){
        return wrappingNode.recieveInferenceSignal(signal);
    }

    /**
     * Get whether the current forward signal is set and valid
     * 
     * @return
     */
    public boolean hasValidForwardSignal(){
        return wrappingNode.hasValidForwardSignal();
    }

    public double[] getWeights(int bitStr){
        return wrappingNode.getWeights(bitStr);
    }

    public double getBias(int bitStr){
        return wrappingNode.getBias(bitStr);
    }

    public void sendForwardSignals(){
        wrappingNode.sendForwardSignals();
    }

    public ArrayList<Outcome> getState(){
        return wrappingNode.getState();
    }

    public void sendErrorsBackwards(ArrayList<Outcome> outcomesAtTime, int timestep){
        wrappingNode.sendErrorsBackwards(null, timestep);
    }

    public void applyErrorSignals(double epsilon){
        wrappingNode.applyErrorSignals(epsilon);
    }

    public void clearSignals(){
        wrappingNode.clearSignals();
    }
}
