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
import com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators.IProbabilityCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.IValueCombinator;
import com.lucasbrown.NetworkTraining.History.IStateRecord;

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of
 * sending a signal out to other connected nodes
 */
public interface INode extends Comparable<INode>, IStateRecord<Outcome>{

    public int getID();

    public String getName();

    public void setName(String name);

    public void setParentNetwork(GraphNetwork network);

    public GraphNetwork getParentNetwork();

    public ActivationFunction getActivationFunction();

    public IValueCombinator getValueCombinator();

    public IProbabilityCombinator getProbabilityCombinator();

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
    public boolean addIncomingConnection(Edge connection);

    public Collection<Edge> getAllIncomingConnections();

    /**
     * Add an outgoing connection to the node
     * 
     * @param connection
     * @return true
     */
    public boolean addOutgoingConnection(Edge connection);

    public Collection<Edge> getAllOutgoingConnections();

    public Optional<Edge> getOutgoingConnectionTo(INode recievingNode);

    public Optional<Edge> getIncomingConnectionFrom(INode sendingNode);

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

    public void setValidForwardSignal(boolean state);

    /**
     * map every incoming node id to its corresponding value and combine.
     * for example, an id of 6 may map to 0b0010 and an id of 2 may map to 0b1000
     * binary_string will thus contain the value 0b1010
     * 
     * @param incomingSignals
     * @return a bit string indicating the weights, bias, and error index to use for
     *         the given set of signals
     */
    public int nodeSetToBinStr(Collection<INode> incomingNodes);

    /**
     * Create an arraylist of arcs from a binary string representation
     * 
     * @param binStr
     * @return
     */
    public ArrayList<Edge> binStrToArcList(int binStr);

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

    public ArrayList<Outcome> getState();
    public void clearSignals();

    public static int CompareNodes(INode n1, INode n2) {
        return n1.getID() - n2.getID();
    }

    public static boolean areNodesEqual(INode n1, INode n2) {
        return n1.getID() == n2.getID();
    }
}
