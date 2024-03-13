package com.lucasbrown.GraphNetwork.Local;

import java.security.InvalidAlgorithmParameterException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.SharedNetworkData;
import com.lucasbrown.GraphNetwork.Local.ReferenceStructure.ReferenceArc;
import com.lucasbrown.NetworkTraining.ApproximationTools.ArrayTools;

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of
 * sending a signal out to other connected nodes
 */
public abstract class Node implements Comparable<Node> {

    protected Random rng = new Random();

    /**
     * All incoming and outgoing node connections.
     */
    protected final ArrayList<Arc> incoming, outgoing;

    /**
     * A unique identifying number for this node.
     */
    protected int id;

    /**
     * A name for this node
     */
    public String name;

    /**
     * The network that this node belongs to
     */
    public GraphNetwork network;

    /**
     * The network hyperparameters
     */
    public SharedNetworkData networkData;

    /**
     * The activation function
     */
    public ActivationFunction activationFunction;

    /**
     * Forward-training signals
     */
    protected ArrayList<Signal> forward, forwardNext;

    /**
     * backward training signals
     */
    protected ArrayList<Signal> backward, backwardNext;

    /**
     * inference signals
     */
    protected ArrayList<Signal> inference, inferenceNext;

    /**
     * Average of all incoming signals
     */
    protected double mergedForwardStrength;

    /**
     * Average of all backward signals
     */
    protected double mergedBackwardStrength;

    /**
     * The signal strength that this node is outputting
     */
    protected double outputStrength;

    protected boolean hasValidForwardSignal;

    public Node(final GraphNetwork network, final SharedNetworkData networkData,
            final ActivationFunction activationFunction, int id) {
        this.id = id;
        name = "Node " + id;
        this.network = Objects.requireNonNull(network);
        this.networkData = Objects.requireNonNull(networkData);
        this.activationFunction = activationFunction;

        incoming = new ArrayList<Arc>();
        outgoing = new ArrayList<Arc>();

        inference = new ArrayList<>();
        inferenceNext = new ArrayList<>();
        forward = new ArrayList<>();
        forwardNext = new ArrayList<>();
        backward = new ArrayList<>();
        backwardNext = new ArrayList<>();
    }

    public int getID()
    {
        return id;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setRandom(Random random) {
        rng = random;
    }

    /**
     * Add an incoming connection to the node
     * 
     * @param connection
     * @return true
     */
    public boolean addIncomingConnection(Arc connection) {
        return incoming.add(connection);
    }

    /**
     * Add an outgoing connection to the node
     * 
     * @param connection
     * @return true
     */
    public boolean addOutgoingConnection(Arc connection) {
        return outgoing.add(connection);
    }

    /**
     * Notify this node of a new incoming forward signal
     * 
     * @param signal
     */
    public void recieveForwardSignal(Signal signal) {
        forwardNext.add(signal);
        network.notifyNodeActivation(this);
    }

    /**
     * Notify this node of a new incoming backward signal
     * 
     * @param signal
     */
    public void recieveBackwardSignal(Signal signal) {
        backwardNext.add(signal);
        network.notifyNodeActivation(this);
    }

    /**
     * Notify this node of a new inference signal
     * 
     * @param signal
     */
    public void recieveInferenceSignal(Signal signal) {
        inferenceNext.add(signal);
        network.notifyNodeActivation(this);
    }

    /**
     * Get whether the current forward signal is set and valid
     * 
     * @return
     */
    public boolean hasValidForwardSignal() {
        return hasValidForwardSignal;
    }

    public double[][] getConnectionParameters() {
        double[][] parameters = new double[outgoing.size()][];
        for (int i = 0; i < parameters.length; i++) {
            parameters[i] = outgoing.get(i).probDist.getParameters();
        }
        return parameters;
    }

    /**
     * Set all next signals to the current signal. Performs additional checks to
     * ensure the current node state is valid. Throws an
     * InvalidAlgorithmParameterException if the state is invalid.
     * 
     * @throws InvalidAlgorithmParameterException
     */
    public void acceptSignals() throws InvalidAlgorithmParameterException {
        if (forwardNext.isEmpty() & backwardNext.isEmpty() & inferenceNext.isEmpty()) {
            throw new InvalidAlgorithmParameterException(
                    "handleIncomingSignals should never be called if no signals have been recieved.");
        }

        if ((!forwardNext.isEmpty() || !backwardNext.isEmpty()) && !inferenceNext.isEmpty()) {
            throw new InvalidAlgorithmParameterException(
                    "Inference and training signals should not be run simultaneously.");
        }

        // Prepare state variables for either inference or training
        if (!inferenceNext.isEmpty()) {
            inference = inferenceNext;
            inferenceNext = new ArrayList<Signal>();
            acceptIncomingForwardSignals(inference);
        } else {
            if (!forwardNext.isEmpty()) {
                forward = forwardNext;
                forwardNext = new ArrayList<Signal>();
                acceptIncomingForwardSignals(forward);
            }
            if (!backwardNext.isEmpty()) {
                backward = backwardNext;
                backwardNext = new ArrayList<Signal>();
                mergedBackwardStrength = getMergedBackwardStrength();
            }
        }

    }

    private double getMergedBackwardStrength() {
        return backward.stream().mapToDouble(Signal::getOutputStrength).average().getAsDouble();
    }

    /**
     * Attempt to send forward and backward signals
     */
    public void sendTrainingSignals() {

        if (!outgoing.isEmpty()) {
            // Send the forward signals and record the cumulative error
            sendForwardSignals();
            hasValidForwardSignal = false;
        }

        if (!incoming.isEmpty()) {
            sendBackwardsSignals();
        }

    }

    public void clearSignals() {
        inference.clear();
        forward.clear();
        backward.clear();
        inferenceNext.clear();
        forwardNext.clear();
        backwardNext.clear();
    }

    @Override
    public String toString() {
        return name + ": " + Double.toString(mergedForwardStrength);
    }

    @Override
    public int hashCode() {
        return id;
    }

    @Override
    public int compareTo(Node o) {
        return id - o.id;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Node))
            return false;
        return id == ((Node) o).id;
    }

    /**
     * 
     * @param node
     * @return whether this node is connected to the provided node
     */
    public boolean doesContainConnection(Node node) {
        return outgoing.stream().map(arc -> (ReferenceArc) arc)
                .anyMatch(connection -> connection.doesMatchNodes(this, node));
    }

    /**
     * Get a list of the node id's of all incoming and outgoing connections
     * 
     * @return a node id list
     */
    public int[] getAllConnectionIDs() {
        return ArrayTools.union(getIncomingConnectionIDs(), getOutgoingConnectionIDs());
    }

    /**
     * Remove all references to the node addressed by {@code node_id}.
     * 
     * @param node_id
     */
    public void removeAllReferencesTo(int node_id){
        removeIncomingConnectionFrom(node_id);
        removeOutgoingConnectionTo(node_id);
    }

    protected abstract void updateWeightsAndBias(double error_derivative);

    /**
     * Compute the merged signal strength of a set of incoming signals
     * 
     * @param incomingSignals
     * @return
     */
    protected abstract double computeMergedSignalStrength(List<Signal> incomingSignals);

    /**
     * Set the state parameters for incoming signal (i.e, either the forward or
     * inference signal)
     * 
     * @param incomingSignals
     */
    protected abstract void acceptIncomingForwardSignals(ArrayList<Signal> incomingSignals);

    /**
     * Attempt to send inference signals to all outgoing connections
     */
    public abstract void sendInferenceSignals();

    /**
     * Send forward signals and record differences of expectation for training
     * 
     * @param
     */
    public abstract void sendForwardSignals();

    /**
     * Send backwards signals and record differences of expectation for training
     * 
     * @param
     */
    public abstract void sendBackwardsSignals();

    /**
     * Apply all changes. Must proceed a call to trainingStep
     */
    public abstract void applyTrainingChanges();

    /**
     * Get a list of the node id's of all incoming connections
     * 
     * @return a node id list
     */
    public abstract int[] getIncomingConnectionIDs();

    /**
     * Get a list of the node id's of all outgoing connections
     * 
     * @return a node id list
     */
    public abstract int[] getOutgoingConnectionIDs();

    /**
     * Remove an incoming connection from the node addressed by {@code node_id}
     * 
     * @param node_id
     * @return true if a connection was remove
     */
    public abstract boolean removeIncomingConnectionFrom(int node_id);

    /**
     * Remove an outgoing connection from the node addressed by {@code node_id}
     * 
     * @param node_id
     * @return true if a connection was remove
     */
    public abstract boolean removeOutgoingConnectionTo(int node_id);

}
