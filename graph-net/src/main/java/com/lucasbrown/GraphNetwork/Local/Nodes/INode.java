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
import com.lucasbrown.NetworkTraining.DataSetTraining.BackwardsSamplingDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.ITrainableDistribution;

public interface INode extends Comparable<INode> {

    public abstract int getID();

    public abstract String getName();

    public abstract void setName(String name);

    public abstract void setParentNetwork(GraphNetwork network);

    public abstract GraphNetwork getParentNetwork();

    public abstract ActivationFunction getActivationFunction();

    public abstract BackwardsSamplingDistribution getOutputDistribution();

    public abstract ITrainableDistribution getSignalChanceDistribution();
 
    /**
     * 
     * @param node
     * @return whether this node is connected to the provided node
     */
    public abstract boolean doesContainConnection(INode node);

    /**
     * Add an incoming connection to the node
     * 
     * @param connection
     * @return true
     */
    public abstract boolean addIncomingConnection(Arc connection);

    public abstract Collection<Arc> getAllIncomingConnections();

    /**
     * Add an outgoing connection to the node
     * 
     * @param connection
     * @return true
     */
    public abstract boolean addOutgoingConnection(Arc connection);

    public abstract Collection<Arc> getAllOutgoingConnections();

    /**
     * Get the arc associated with the transfer from this node to the given
     * recieving node
     * 
     * @param recievingNode
     * @return The arc if present, otherwise null
     */
    public abstract Optional<Arc> getOutgoingConnectionTo(INode recievingNode);

    /**
     * Get the arc associated with the transfer from the given sending node to this
     * node
     * 
     * 
     * @param recievingNode
     * @return The arc if present, otherwise null
     */
    public abstract Optional<Arc> getIncomingConnectionFrom(INode sendingNode);

    /**
     * Notify this node of a new incoming forward signal
     * 
     * @param signal
     */
    public abstract void recieveForwardSignal(Signal signal);

    /**
     * Notify this node of a new incoming backward signal
     * 
     * @param signal
     */
    public abstract void recieveBackwardSignal(Signal signal);

    /**
     * Notify this node of a new inference signal
     * 
     * @param signal
     */
    public abstract void recieveInferenceSignal(Signal signal);

    /**
     * Get whether the current forward signal is set and valid
     * 
     * @return
     */
    public abstract boolean hasValidForwardSignal();

    /**
     * Get whether the current forward signal is set and valid
     * 
     * @return
     */
    public abstract void setValidForwardSignal(boolean state);

    public abstract double[] getWeights(int bitStr);

    public abstract double getBias(int bitStr);

    /**
     * Set all next signals to the current signal. Performs additional checks to
     * ensure the current node state is valid. Throws an
     * InvalidAlgorithmParameterException if the state is invalid.
     * 
     * @throws InvalidAlgorithmParameterException
     */
    public abstract void acceptSignals() throws InvalidAlgorithmParameterException;

    public abstract void sendTrainingSignals();

    /**
     * Send forward signals and record differences of expectation for training
     * 
     * @param
     */
    public abstract void sendForwardSignals();

    public abstract ArrayList<Outcome> getState();

    public abstract void sendErrorsBackwards(ArrayList<Outcome> outcomesAtTime, int timestep);

    public abstract void applyErrorSignals(double epsilon, List<ArrayList<Outcome>> allOutcomes);

    public abstract void applyDistributionUpdate();

    public abstract void applyFilterUpdate();

    public abstract void clearSignals();

    public static int CompareNodes(INode n1, INode n2) {
        return n1.getID() - n2.getID();
    }

    public static boolean areNodesEqual(INode n1, INode n2) {
        return n1.getID() == n2.getID();
    }
}
