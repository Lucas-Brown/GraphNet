package com.lucasbrown.GraphNetwork.Local.Nodes;

import java.security.InvalidAlgorithmParameterException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Optional;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Signal;

public interface INode extends Comparable<INode> {

    public int getID();

    public String getName();

    public void setName(String name);

    public void setParentNetwork(GraphNetwork network);

    public GraphNetwork getParentNetwork();

    public ActivationFunction getActivationFunction();

    /**
     * 
     * @param node
     * @return whether this node is connected to the provided node
     */
    public boolean doesContainConnection(INode node);

    /**
     * Add an incoming connection to the node
     * 
     * @param connection
     * @return true
     */
    public boolean addIncomingConnection(Arc connection);

    public Collection<Arc> getAllIncomingConnections();

    /**
     * Add an outgoing connection to the node
     * 
     * @param connection
     * @return true
     */
    public boolean addOutgoingConnection(Arc connection);

    public Collection<Arc> getAllOutgoingConnections();

    /**
     * Get the arc associated with the transfer from this node to the given
     * recieving node
     * 
     * @param recievingNode
     * @return The arc if present, otherwise null
     */
    public Optional<Arc> getOutgoingConnectionTo(INode recievingNode);

    /**
     * Get the arc associated with the transfer from the given sending node to this
     * node
     * 
     * 
     * @param recievingNode
     * @return The arc if present, otherwise null
     */
    public Optional<Arc> getIncomingConnectionFrom(INode sendingNode);

    /**
     * Notify this node of a new incoming forward signal
     * 
     * @param signal
     */
    public void recieveForwardSignal(Signal signal);

    /**
     * Get whether the current forward signal is set and valid
     * 
     * @return
     */
    public boolean hasValidForwardSignal();

    /**
     * Get whether the current forward signal is set and valid
     * 
     * @return
     */
    public void setValidForwardSignal(boolean state);

    /**
     * Set all next signals to the current signal. Performs additional checks to
     * ensure the current node state is valid. Throws an
     * InvalidAlgorithmParameterException if the state is invalid.
     * 
     * @throws InvalidAlgorithmParameterException
     */
    public void acceptSignals() throws InvalidAlgorithmParameterException;

    /**
     * Send forward signals and record differences of expectation for training
     * 
     * @param
     */
    public void sendForwardSignals();

    public void clearSignals();

    /**
     * Get a view of the internal state of this node
     * 
     * @return
     */
    public ArrayList<Outcome> getState();

    public double computeMergedSignalStrength(Collection<Signal> incomingSignals, int binary_string);

    public static int CompareNodes(INode n1, INode n2) {
        return n1.getID() - n2.getID();
    }

    public static boolean areNodesEqual(INode n1, INode n2) {
        return n1.getID() == n2.getID();
    }
}
