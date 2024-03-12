package com.lucasbrown.GraphNetwork.Local.DataStructure;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import com.lucasbrown.GraphNetwork.Global.DataGraphNetwork;
import com.lucasbrown.GraphNetwork.Global.SharedNetworkData;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.FilterDistribution;
import com.lucasbrown.GraphNetwork.Local.ICopyable;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.GraphNetwork.Local.Signal;
import com.lucasbrown.GraphNetwork.Local.ReferenceStructure.ReferenceNode;

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of
 * sending a signal out to other connected nodes
 */
public class DataNode extends Node implements ICopyable<DataNode> {

    /**
     * Maps every combination of inputs to corresponding weights and bias
     */
    protected final HashMap<HashSet<Integer>, WeightsAndBias> nodeSetToWeightsAndBias;

    /**
     * The error derivative at the given time step
     */
    private double error;

    /**
     * The corresponding weights and bias for the set of forward signals
     */
    private WeightsAndBias weightsAndBias;

    /**
     * The node set representing the forward signals
     */
    private HashSet<Integer> forwardNodeSet;

    /**
     * The node set representing the backward signals
     */
    private HashSet<Integer> backwardNodeSet;

    private int total_parameters = 0;

    public DataNode(final DataGraphNetwork network, final SharedNetworkData networkData,
            final ActivationFunction activationFunction, int id) {
        super(network, networkData, activationFunction, id);

        nodeSetToWeightsAndBias = new HashMap<HashSet<Integer>, WeightsAndBias>();
        nodeSetToWeightsAndBias.put(new HashSet<Integer>(0), new WeightsAndBias(0, new double[0]));
    }

    public DataNode(DataNode toCopy)
    {
        super(toCopy.network, toCopy.networkData, toCopy.activationFunction, toCopy.id);
        nodeSetToWeightsAndBias = new HashMap<>(toCopy.nodeSetToWeightsAndBias);
    }

    @Override
    public boolean addIncomingConnection(Arc connection) {
        appendWeightsAndBiases(connection.getSendingID());
        return super.addIncomingConnection(connection);
    }

    public boolean removeIncomingConnection(Node sendingNode) {
        removeWeightsAndBiases(sendingNode);
        return incoming.removeIf(arc -> arc.getSendingID() == sendingNode.id);
    }

    /**
     * Add an outgoing connection to the node
     * 
     * @param connection
     * @return true
     */
    public boolean addOutgoingConnection(Arc connection) {
        total_parameters += connection.probDist.getParameters().length;
        return outgoing.add(connection);
    }

    
    /**
     * Add an outgoing connection to the node
     * 
     * @param connection
     * @return true
     */
    public boolean removeOutgoingConnection(Node recievingNode) {
        total_parameters -= incoming.stream().filter(arc -> arc.getSendingID() == recievingNode.id)
                .mapToInt(arc -> arc.probDist.getParameters().length).count();
        return outgoing.removeIf(arc -> arc.getRecievingID() == recievingNode.id);
    }

    /**
     * Creates a new set of weights and biases for the incoming ndoe
     */
    private void appendWeightsAndBiases(int sendingNodeID) {
        final HashMap<HashSet<Integer>, WeightsAndBias> copy_nodeSetToWeightsAndBias = new HashMap<>(
                nodeSetToWeightsAndBias);

        int size = copy_nodeSetToWeightsAndBias.size();
        total_parameters += weightsAndBiasesCount(size+1)-weightsAndBiasesCount(size);
        for (HashSet<Integer> intSet : copy_nodeSetToWeightsAndBias.keySet()) {
            HashSet<Integer> copy_intSet = new HashSet<>(intSet);
            copy_intSet.add(sendingNodeID);
            nodeSetToWeightsAndBias.put(copy_intSet, new WeightsAndBias(rng, copy_intSet.size()));
        }
    }

    /**
     * Remove all weights and biases associated with the incoming node
     */
    private void removeWeightsAndBiases(Node sendingNode) {
        final int node_id = sendingNode.id;
        int size = nodeSetToWeightsAndBias.size();
        total_parameters += weightsAndBiasesCount(size-1)-weightsAndBiasesCount(size);
        nodeSetToWeightsAndBias.keySet().removeIf(intSet -> intSet.contains(node_id));
    }

    private int weightsAndBiasesCount(int n_connections)
    {
        if(n_connections == 0) 
        {
            return 0;
        }
        else
        {
            return (n_connections + 2) * (0b1 << (n_connections-1));
        }
    }

    @Override
    protected void updateWeightsAndBias(double error_derivative) {

        /* 
        // compute delta to update the weights and bias
        double delta = -networkData.getEpsilon() * error_derivative;
        assert Double.isFinite(delta);

        WeightsAndBias wAb = biases[binary_string] += delta;

        for (int weight_idx = 0; weight_idx < weights[binary_string].length; weight_idx++) {
            weights[binary_string][weight_idx] += delta * forward.get(weight_idx).strength;
        }
        */
    }

    public int getTotalNumberOfParameters()
    {
        return total_parameters;
    }

    public int getNumberOfBiases()
    {
        return nodeSetToWeightsAndBias.size();
    }

    public int getNumberOfWeightAndBiasParameters()
    {
        return weightsAndBiasesCount(nodeSetToWeightsAndBias.size());
    }


    public double[][] getWeights() {
        return nodeSetToWeightsAndBias.entrySet()
                .stream()
                .sorted(new IntSetComparator())
                .map(entry -> entry.getValue().weights)
                .toArray(double[][]::new);
    }

    public double[] getBiases() {
        return nodeSetToWeightsAndBias.entrySet()
                .stream()
                .sorted(new IntSetComparator())
                .mapToDouble(entry -> entry.getValue().bias)
                .toArray();
    }

    public WeightsAndBias getWeightsAndBias(int bin_str)
    {
        return nodeSetToWeightsAndBias.get(binStrToHashSet(bin_str));
    }

    public Collection<WeightsAndBias> getWeightsAndBiases()
    {
        return nodeSetToWeightsAndBias.values();
    }

    public FilterDistribution[] getFilters()
    {
        return outgoing.stream()
                .sorted((arc1, arc2) -> arc1.getRecievingID() - arc2.getRecievingID())
                .map(arc -> arc.probDist)
                .toArray(FilterDistribution[]::new);
    }

    public double[][] getConnectionParameters() {
        double[][] parameters = new double[outgoing.size()][];
        for (int i = 0; i < parameters.length; i++) {
            parameters[i] = outgoing.get(i).probDist.getParameters();
        }
        return parameters;
    }

    /**
     * Get all outgoing connections as a boolean array
     * 
     * @return
     */
    public boolean[] getConnectivity(int size) {
        boolean[] connectivity = new boolean[size];
        for (Arc arc : outgoing) {
            connectivity[arc.getRecievingID()] = true;
        }
        return connectivity;
    }

    /**
     * Get the weight of a node through its node ID and the bit string of the
     * corresponding combination of inputs
     * 
     * @param bitStr
     * @param nodeId
     */
    /*
     * public double getWeightOfNode(int bitStr, int nodeId) {
     * int nodeBitmask = orderedIDMap.get(nodeId);
     * assert (bitStr & nodeBitmask) > 0 :
     * "bit string does not contain the index of the provided node ID";
     * 
     * // find the number of ocurrences of 1's up to the index of the node
     * int nodeIdx = 0;
     * int bitStrShifted = bitStr;
     * while (nodeBitmask > 0b1) {
     * if ((bitStrShifted & 0b1) == 1) {
     * nodeIdx++;
     * }
     * bitStrShifted = bitStrShifted >> 1;
     * nodeBitmask = nodeBitmask >> 1;
     * }
     * 
     * return weights[bitStr][nodeIdx];
     * }
     */

    /**
     * Compute the merged signal strength of a set of incoming signals
     * 
     * @param incomingSignals
     * @return
     */
    @Override
    protected double computeMergedSignalStrength(List<Signal> incomingSignals) {
        double strength = IntStream.range(0, weightsAndBias.weights.length)
                .mapToDouble(i -> weightsAndBias.weights[i] * incomingSignals.get(i).strength)
                .sum();

        strength += weightsAndBias.bias;
        return strength;
    }

    /**
     * Set the state parameters for incoming signal (i.e, either the forward or
     * inference signal)
     * 
     * @param incomingSignals
     */
    protected void acceptIncomingForwardSignals(ArrayList<Signal> incomingSignals) {
        if (incomingSignals.size() == 0)
            return;
        hasValidForwardSignal = true;

        // Convert the incoming signal list to a hashset of id's
        forwardNodeSet = incomingSignals.stream().map(signal -> signal.sendingNode.id)
                .collect(Collectors.toCollection(HashSet<Integer>::new));
        
        weightsAndBias = nodeSetToWeightsAndBias.get(forwardNodeSet);

        // sorting by id to ensure that the weights are applied to the correct
        // node/signal
        incomingSignals.sort((s1, s2) -> Integer.compare(s1.recievingNode.id, s2.recievingNode.id));

        mergedForwardStrength = computeMergedSignalStrength(incomingSignals);
        assert Double.isFinite(mergedForwardStrength);
        outputStrength = activationFunction.activator(mergedForwardStrength);
    }

    /**
     * Attempt to send inference signals to all outgoing connections
     */
    @Override
    public void sendInferenceSignals() {
        for (Arc connection : outgoing) {
            // roll and send a signal if successful
            if (connection.rollFilter(mergedForwardStrength)) {
                connection.sendInferenceSignal(outputStrength);
            }
        }
        hasValidForwardSignal = false;
    }

    /**
     * Apply all changes. Must proceed a call to trainingStep
     */
    @Override
    public void applyTrainingChanges() {

        // update weights and biases to reinforce forward signals
        // if the error == NAN then this node failed to send a signal to the next
        if (!forward.isEmpty() && !Double.isNaN(error))
            updateWeightsAndBias(error);

        // reinforce backward signals
        if (!backward.isEmpty()) {
            // ArrayList<Arc> arcs = binStrToArcList(backwardsBinStr);
            // arcs.forEach(arc -> arc.probDist.applyAdjustments());
        }
    }

    /**
     * Send forward signals and record differences of expectation for training
     * 
     * @param
     */
    public void sendForwardSignals() {
        int count = 0;
        double[] expectedValues = new double[outgoing.size()];
        for (Arc connection : outgoing) {
            if (connection.rollFilter(mergedForwardStrength)) {
                connection.sendForwardSignal(outputStrength);
                expectedValues[count++] = connection.probDist.getMean();
            }
        }

        error = errorDerivativeOfOutput(expectedValues, count);
    }

    private double errorDerivativeOfOutput(double[] expectedValues, int count) {
        double error = 0;
        for (int i = 0; i < count; i++) {
            error += networkData.errorFunc.error_derivative(mergedForwardStrength, expectedValues[i]);
        }
        return error / count;
    }

    private HashSet<Integer> binStrToHashSet(int binStr) {
        HashSet<Integer> intSet = new HashSet<>();
        for (int i = 0; i < incoming.size(); i++) {
            if ((binStr & 0b1) == 1) {
                intSet.add(i);
            }
            binStr = binStr >> 1;
        }
        return intSet;
    }

    /**
     * Send backwards signals and record differences of expectation for training
     * 
     * @param
     */
    @Override
    public void sendBackwardsSignals() {
        return;
        /*
         * // Select the backward signal combination
         * Convolution[] convolutions = getReverseOutcomes();
         * double[] densityWeights = evaluateConvolutions(convolutions);
         * backwardsBinStr = selectReverseOutcome(densityWeights) + 1;
         * 
         * // Sample signal strengths from the selected distribution
         * double[] sample = convolutions[backwardsBinStr -
         * 1].sample(mergedBackwardStrength);
         * 
         * // Send the backward signals
         * ArrayList<Arc> arcs = binStrToArcList(backwardsBinStr);
         * for (int i = 0; i < sample.length; i++) {
         * Arc arc_i = arcs.get(i);
         * double sample_i = sample[i];
         * double sample_inverse = arc_i.recieving.activationFunction.inverse(sample_i);
         * arc_i.sendBackwardSignal(sample_inverse); // send signal backwards
         * arc_i.probDist.prepareReinforcement(sample_inverse); // prepare to reinforce
         * the distribution
         * }
         */
    }

    /*
     * public Convolution[] getReverseOutcomes() {
     * // Loop over all possible incoming signal combinations and record the value
     * of
     * // their convolution
     * int n_choices = 0b1 << incoming.size();
     * Convolution[] convolutions = new Convolution[n_choices - 1];
     * for (int binStr = 1; binStr < n_choices; binStr++) {
     * // get the arcs corresponding to this bit string
     * HashSet<Integer> intSet = binStrToHashSet(binStr);
     * 
     * // Seperate the arcs into their distributions and activation functions
     * ArrayList<FilterDistribution> distributions = intSet.stream()
     * .map(arc -> arc.probDist)
     * .collect(Collectors.toCollection(ArrayList<FilterDistribution>::new));
     * 
     * ArrayList<ActivationFunction> activators = arcs.stream()
     * .map(arc -> arc.sending.activationFunction)
     * .collect(Collectors.toCollection(ArrayList<ActivationFunction>::new));
     * 
     * // Get the weights of the corresponding arcs
     * double[] weights = getWeights(binStr);
     * 
     * // Get the probability density
     * convolutions[binStr - 1] = new Convolution(distributions, activators,
     * weights);
     * }
     * 
     * return convolutions;
     * }
     * 
     * public double[] evaluateConvolutions(Convolution[] convolutions) {
     * double[] densityWeights = new double[convolutions.length];
     * for (int binStr = 1; binStr <= convolutions.length; binStr++) {
     * // Shift the signal strength by the bias
     * double shiftedStrength = mergedBackwardStrength - biases[binStr];
     * 
     * densityWeights[binStr - 1] = convolutions[binStr -
     * 1].convolve(shiftedStrength);
     * assert Double.isFinite(densityWeights[binStr - 1]);
     * assert densityWeights[binStr - 1] >= 0;
     * }
     * return densityWeights;
     * }
     */

    /**
     * Select an outcome each with a given weight
     * 
     * @param densityWeights
     * @return
     */
    public int selectReverseOutcome(double[] densityWeights) {
        // get the sum of all weights
        double total = 0;
        for (double d : densityWeights) {
            total += d;
        }

        // Roll a number between 0 and the total
        double roll = rng.nextDouble() * total;

        // Select the first index whose partial sum is over the roll
        double sum = 0;
        for (int i = 0; i < densityWeights.length; i++) {
            sum += densityWeights[i];
            if (sum >= roll) {
                return i;
            }
        }
        return -1; // This should never happen
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ReferenceNode))
            return false;
        return id == ((ReferenceNode) o).id;
    }

    
    @Override
    public DataNode copy() {
        return new DataNode(this);
    }

    public class WeightsAndBias {
        public double bias;
        public double[] weights;

        public WeightsAndBias(double bias, double[] weights) {
            this.bias = bias;
            this.weights = weights;
        }

        public WeightsAndBias(Random rng, int size) {
            bias = rng.nextDouble();
            weights = DoubleStream.generate(rng::nextDouble).limit(size).toArray();
        }
    }

    private class IntSetComparator implements Comparator<Entry<HashSet<Integer>, WeightsAndBias>> {

        @Override
        public int compare(Entry<HashSet<Integer>, WeightsAndBias> set1, Entry<HashSet<Integer>, WeightsAndBias> set2) {
            ArrayList<Integer> arr1 = new ArrayList<>(set1.getKey());
            ArrayList<Integer> arr2 = new ArrayList<>(set2.getKey());
            arr1.sort(Comparator.naturalOrder());
            arr2.sort(Comparator.naturalOrder());

            int min = Math.min(arr1.size(), arr2.size());
            for (int i = 0; i < min; i++) {
                int comp = Integer.compare(arr1.get(i), arr2.get(i));
                if (comp != 0) {
                    return comp;
                }

            }

            // if the two are equal, compare base on length
            return arr1.size() - arr2.size();
        }

    }

    @Override
    public int[] getIncomingConnectionIDs() {
        return incoming.stream().mapToInt(arc -> arc.getSendingID()).toArray();
    }

    @Override
    public int[] getOutgoingConnectionIDs() {
        return outgoing.stream().mapToInt(arc -> arc.getRecievingID()).toArray();
    }

    @Override
    public boolean removeIncomingConnectionFrom(int node_id) {
        incoming.removeIf(arc -> arc.getSendingID() == node_id);
        return nodeSetToWeightsAndBias.keySet().removeIf(node_set -> node_set.contains(node_id));
    }

    @Override
    public boolean removeOutgoingConnectionTo(int node_id) {
        return outgoing.removeIf(arc -> arc.getRecievingID() == node_id);
    }




}
