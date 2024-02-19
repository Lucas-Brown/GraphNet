package com.lucasbrown.GraphNetwork.Local;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.stream.IntStream;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.SharedNetworkData;
import com.lucasbrown.GraphNetwork.Global.Signal;

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of
 * sending a signal out to other connected nodes
 */
public class Node implements Comparable<Node> {

    /**
     * The coutner is used to give each node a unique ID
     */
    private static int ID_COUNTER = 0;

    /**
     * A unique identifying number for this node.
     */
    public final int id;

    /**
     * A name for this node
     */
    public String name;

    /**
     * Stores whether this node is active in the current step
     */
    private boolean isActive;

    /**
     * The network that this node belongs to
     */
    public final GraphNetwork network;

    /**
     * The network hyperparameters
     */
    public final SharedNetworkData networkData;

    /**
     * The activation function
     */
    public final ActivationFunction activationFunction;

    /**
     * All incoming and outgoing node connections.
     */
    protected final ArrayList<Arc> incoming, outgoing;

    /**
     * All incoming signals
     */
    protected ArrayList<Signal> incomingSignals;

    /**
     * All outgoing signals
     */
    protected ArrayList<Signal> outgoingSignals;

    /**
     * All signals that have been processed
     */
    private ArrayList<Signal> acceptedSignals;

    /**
     * All error signals
     */
    protected final ArrayList<Signal> errorSignals;

    /**
     * Maps all incoming node ID's to an int from 0 to the number of incoming nodes
     * -1
     */
    protected final HashMap<Integer, Integer> orderedIDMap;

    /**
     * Each possible combinations of inputs has a corresponding unique set of
     * weights and biases
     * both grow exponentially, which is bad, but every node should have relatively
     * few connections
     */
    private double[][] weights;
    private double[] biases;

    /**
     * The sum of all incoming signals
     */
    protected double mergedSignalStrength;

    /**
     * The signal strength that this node is outputting
     */
    protected double outputStrength;

    /**
     * The binary representation of the currently incoming signals
     */
    private int binary_string;

    public Node(final GraphNetwork network, final SharedNetworkData networkData,
            final ActivationFunction activationFunction) {
        id = ID_COUNTER++;
        name = "Node " + id;
        this.network = Objects.requireNonNull(network);
        this.networkData = Objects.requireNonNull(networkData);
        this.activationFunction = activationFunction;
        incoming = new ArrayList<Arc>();
        outgoing = new ArrayList<Arc>();
        incomingSignals = new ArrayList<>();
        outgoingSignals = new ArrayList<>();
        acceptedSignals = new ArrayList<>();
        errorSignals = new ArrayList<>();
        orderedIDMap = new HashMap<>();
        weights = new double[1][1];
        biases = new double[1];
        weights[0] = new double[0];
        isActive = false;
    }

    public void setName(String name) {
        this.name = name;
    }

    /**
     * Get whether this node is active (i.e. has a valid value)
     * 
     * @return
     */
    public boolean isActive() {
        return isActive;
    }

    public void activate() {
        isActive = true;
    }

    public void deactivate() {
        isActive = false;
        outgoingSignals.clear();
        acceptedSignals.clear();
    }

    /**
     * 
     * @param node
     * @return whether this node is connected to the provided node
     */
    public boolean doesContainConnection(Node node) {
        return outgoing.stream().anyMatch(connection -> connection.doesMatchNodes(this, node));
    }

    /**
     * Get the arc associated with the transfer from this node to the given
     * recieving node
     * 
     * @param recievingNode
     * @return The arc if present, otherwise null
     */
    public Arc getArc(Node recievingNode) {
        for (Arc arc : outgoing) {
            if (arc.doesMatchNodes(this, recievingNode)) {
                return arc;
            }
        }
        return null;
    }

    /**
     * Add an incoming connection to the node
     * 
     * @param connection
     * @return true
     */
    public boolean addIncomingConnection(Arc connection) {
        orderedIDMap.put(connection.sending.id, 1 << orderedIDMap.size());
        appendWeightsAndBiases();
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
     * Notify this node of a new incoming signal
     * 
     * @param signal The value of the incoming signal
     */
    void recieveSignal(Signal signal) {
        incomingSignals.add(signal);
    }

    /**
     * Notify this node that it has recieved an error signal
     * 
     * @param error
     */
    void recieveErrorSignal(Signal signal) {
        errorSignals.add(signal);
    }

    /**
     * Notify this node of a new incoming signal
     * 
     * @param signal The value of the incoming signal
     */
    void transmittingSignal(Signal signal) {
        outgoingSignals.add(signal);
    }

    /**
     * Adds another layer of depth to the weights and biases hyper array
     */
    private void appendWeightsAndBiases() {
        Random rand = new Random();
        final int old_size = biases.length;
        final int new_size = old_size * 2;

        // the first half doesn't need to be changed
        biases = Arrays.copyOf(biases, new_size);
        weights = Arrays.copyOf(weights, new_size);

        // the second half needs entirely new data
        for (int i = old_size; i < new_size; i++) {
            biases[i] = rand.nextDouble();

            // populate the weights array
            int count = weights[i - old_size].length + 1;
            weights[i] = new double[count];
            for (int j = 0; j < count; j++) {
                weights[i][j] = rand.nextDouble();
            }
        }
    }

    private void updateWeightsAndBias(double error_derivative) {

        // compute delta to update the weights and bias
        double delta = -networkData.getEpsilon() * error_derivative;
        biases[binary_string] += delta;

        for (int weight_idx = 0; weight_idx < weights[binary_string].length; weight_idx++) {
            weights[binary_string][weight_idx] += delta * incomingSignals.get(weight_idx).strength;
        }
    }

    public double[] getWeights(int bitStr) {
        return weights[bitStr].clone(); // A shallow clone is okay here
    }

    /**
     * Get the weight of a node through its node ID and the bit string of the
     * corresponding combination of inputs
     * 
     * @param bitStr
     * @param nodeId
     */
    public double getWeightOfNode(int bitStr, int nodeId) {
        int nodeBitmask = orderedIDMap.get(nodeId);
        assert (bitStr & nodeBitmask) > 0 : "bit string does not contain the index of the provided node ID";

        // find the number of ocurrences of 1's up to the index of the node
        int nodeIdx = 0;
        int bitStrShifted = bitStr;
        while (nodeBitmask > 0b1) {
            if ((bitStrShifted & 0b1) == 1) {
                nodeIdx++;
            }
            bitStrShifted = bitStrShifted >> 1;
            nodeBitmask = nodeBitmask >> 1;
        }

        return weights[bitStr][nodeIdx];
    }

    /**
     * map every incoming node id to its corresponding value and combine.
     * for example, an id of 6 may map to 0b0010 and an id of 2 may map to 0b1000
     * binary_string will thus contain the value 0b1010
     * 
     * @param incomingSignals
     * @return a bit string indicating the weights, bias, and error index to use for
     *         the given set of signals
     */
    public int nodeSetToBinStr(List<Node> incomingNodes) {
        return incomingNodes.stream()
                .mapToInt(node -> orderedIDMap.get(node.id))
                .reduce(0, (result, id_bit) -> result |= id_bit); // effectively the same as a sum in this case
    }

    /**
     * Compute the merged signal strength of a set of incoming signals
     * 
     * @param incomingSignals
     * @return
     */
    private double computeMergedSignalStrength(List<Signal> incomingSignals) {

        double[] input_weights = weights[binary_string];

        double strength = IntStream.range(0, input_weights.length)
                .mapToDouble(i -> input_weights[i] * incomingSignals.get(i).strength)
                .sum();

        strength += biases[binary_string];

        return strength;
    }

    private double computeMergedSignalStrength() {
        return computeMergedSignalStrength(incomingSignals);
    }

    /**
     * Handle all incoming signals and store the resulting strength
     */
    public void acceptSignals() {
        assert !incomingSignals.isEmpty() || !errorSignals.isEmpty()
                : "handleIncomingSignals should never be called if no signals have been recieved.";

        // Compute the binary string of the incoming signals
        binary_string = nodeSetToBinStr(incomingSignals.stream().map(Signal::getSendingNode).toList());

        acceptIncomingSignals();
    }

    private void acceptIncomingSignals() {
        incomingSignals.sort((s1, s2) -> Integer.compare(s1.recievingNode.id, s2.recievingNode.id)); // sorting by id
                                                                                                     // ensure that the
                                                                                                     // weights are
                                                                                                     // applied to the
                                                                                                     // correct
                                                                                                     // node/signal

        mergedSignalStrength = computeMergedSignalStrength();
        outputStrength = activationFunction.activator(mergedSignalStrength);

        acceptedSignals = incomingSignals;
        incomingSignals = new ArrayList<>();
    }

    private void acceptErrorSignals() {
        errorSignals.sort((s1, s2) -> Integer.compare(s1.recievingNode.id, s2.recievingNode.id)); // sorting by id
                                                                                                  // ensure that the
                                                                                                  // weights are applied
                                                                                                  // to the correct
                                                                                                  // node/signal

        mergedSignalStrength = computeMergedSignalStrength();
        outputStrength = activationFunction.activator(mergedSignalStrength);

        acceptedSignals = incomingSignals;
        incomingSignals = new ArrayList<>();
    }

    /**
     * Attempt to send a signal out to every outward connecting neuron
     */
    public void attemptSendOutgoingSignals() {
        // double factor = 1f;
        // final double decay = networkData.getLikelyhoodDecay();

        // Collections.shuffle(outgoing); // shuffle to ensure no connection has an
        // order-dependent advantage

        int count = 0;
        double[] expectedValues = new double[outgoing.size()];
        for (Arc connection : outgoing) {
            if (connection.sendSignal(mergedSignalStrength, outputStrength) != null) {
                expectedValues[count++] = connection.probDist.getMeanValue();
            }
        }
        ;
        updateWeightsAndBias(errorDerivativeOfOutput(expectedValues, count));
    }

    private double errorDerivativeOfOutput(double[] expectedValues, int count) {
        double error = 0;
        for (int i = 0; i < count; i++) {
            error += networkData.errorFunc.error_derivative(mergedSignalStrength, expectedValues[i]);
        }
        return error / count;
    }

    @Override
    public String toString() {
        return name + ": " + Double.toString(outputStrength);
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

}