package com.lucasbrown.GraphNetwork.Local.Nodes;

import java.security.InvalidAlgorithmParameterException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.lucasbrown.GraphNetwork.Distributions.FilterDistribution;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.SharedNetworkData;
import com.lucasbrown.NetworkTraining.ApproximationTools.ArrayTools;
import com.lucasbrown.NetworkTraining.ApproximationTools.Convolution.FilterDistributionConvolution;

public interface INode extends Comparable<INode> {

    public abstract int getID();
    public abstract void setName(String name);

    /**
     * 
     * @param node
     * @return whether this node is connected to the provided node
     */
    public abstract boolean doesContainConnection(INode node);

    /**
     * Get the arc associated with the transfer from this node to the given
     * recieving node
     * 
     * @param recievingNode
     * @return The arc if present, otherwise null
     */
    public abstract Arc getArc(INode recievingNode);

    /**
     * Add an incoming connection to the node
     * 
     * @param connection
     * @return true
     */
    public abstract boolean addIncomingConnection(Arc connection);

    /**
     * Add an outgoing connection to the node
     * 
     * @param connection
     * @return true
     */
    public abstract boolean addOutgoingConnection(Arc connection);

    public abstract Optional<Arc> getOutgoingConnectionTo(Node recievingNode);

    /**
     * Notify this node of a new incoming forward signal
     * 
     * @param signal
     */
    public abstract void recieveForwardSignal(Signal signal);

    public abstract void recieveError(int timestep, int key, double error);

    public abstract Double getError(int timestep, int key);

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

    /**
     * Send forward signals and record differences of expectation for training
     * 
     * @param
     */
    public abstract void sendForwardSignals();


    public abstract ArrayList<Outcome> getState();

    public abstract void sendErrorsBackwards(ArrayList<Outcome> outcomesAtTime, int timestep);

    public abstract void applyErrorSignals(double epsilon);

    public abstract void clearSignals();
}
