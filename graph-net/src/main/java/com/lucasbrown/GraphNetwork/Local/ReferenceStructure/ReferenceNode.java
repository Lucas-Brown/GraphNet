package com.lucasbrown.GraphNetwork.Local.ReferenceStructure;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.lucasbrown.GraphNetwork.Global.ReferenceGraphNetwork;
import com.lucasbrown.GraphNetwork.Global.SharedNetworkData;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.FilterDistribution;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.GraphNetwork.Local.Signal;
import com.lucasbrown.NetworkTraining.ApproximationTools.Convolution;

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of
 * sending a signal out to other connected nodes
 */
public class ReferenceNode extends Node {

    /**
     * The coutner is used to give each node a unique ID
     */
    private static int ID_COUNTER = 0;

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
    protected double[][] weights;
    protected double[] biases;

    /**
     * The binary representation of the currently incoming signals
     */
    private int binary_string;

    /**
     * The error derivative at the given time step
     */
    private double error;

    /**
     * The binary string representation of the incoming arcs being sent a backwards
     * signal
     */
    private int backwardsBinStr;

    public ReferenceNode(final ReferenceGraphNetwork network, final SharedNetworkData networkData,
            final ActivationFunction activationFunction) {
        super(network, networkData, activationFunction, ID_COUNTER++);

        orderedIDMap = new HashMap<Integer, Integer>();

        weights = new double[1][1];
        biases = new double[1];
        weights[0] = new double[0];

    }

    @Override
    public boolean addIncomingConnection(Arc connection) {
        orderedIDMap.put(connection.getSendingID(), 1 << orderedIDMap.size());
        appendWeightsAndBiases();
        return super.addIncomingConnection(connection);
    }

    /**
     * Add an outgoing connection to the node
     * 
     * @param connection
     * @return true
     */
    public boolean addOutgoingConnection(ReferenceArc connection) {
        return outgoing.add(connection);
    }

    
    /**
     * Get the arc associated with the transfer from this node to the given
     * recieving node
     * 
     * @param recievingNode
     * @return The arc if present, otherwise null
     */
    public ReferenceArc getArc(Node recievingNode) {
        for (Arc arc : outgoing) {
            ReferenceArc arc_ref = (ReferenceArc) arc;
            if (arc_ref.doesMatchNodes(this, recievingNode)) {
                return arc_ref;
            }
        }
        return null;
    }

    /**
     * Manually set all the connections of the network.
     * 
     * @param parameters
     * @param weights
     * @param biases
     */
    /*
     * public void setConnectionsData(double[][] parameters, double[][] weights,
     * double[] biases) {
     * final int n_choices = this.weights.length;
     * if (weights.length != n_choices) {
     * throw new InvalidParameterException(
     * "Size of weights does not match.");
     * }
     * if (biases.length != n_choices) {
     * throw new InvalidParameterException(
     * "Size of biases does not match.");
     * }
     * if(outgoing.size() != parameters.length)
     * {
     * throw new InvalidParameterException(
     * "Size of parameters does not match.");
     * }
     * 
     * this.weights = weights;
     * this.biases = biases;
     * for (int i = 0; i < outgoing.size(); i++) {
     * outgoing.get(i).probDist.setParameters(parameters[i]);
     * }
     * }
     */

    /**
     * Adds another layer of depth to the weights and biases hyper array
     */
    private void appendWeightsAndBiases() {
        final int old_size = biases.length;
        final int new_size = old_size * 2;

        // the first half doesn't need to be changed
        biases = Arrays.copyOf(biases, new_size);
        weights = Arrays.copyOf(weights, new_size);

        // the second half needs entirely new data
        for (int i = old_size; i < new_size; i++) {
            biases[i] = rng.nextDouble();

            // populate the weights array
            int count = weights[i - old_size].length + 1;
            weights[i] = new double[count];
            for (int j = 0; j < count; j++) {
                weights[i][j] = rng.nextDouble();
            }
        }
    }

    /*
     * private void removeLastWeightAndBias(){
     * biases = Arrays.copyOf(biases, biases.length/2);
     * weights = Arrays.copyOf(weights, weights.length/2);
     * }
     */

    @Override
    protected void updateWeightsAndBias(double error_derivative) {

        // compute delta to update the weights and bias
        double delta = -networkData.getEpsilon() * error_derivative;
        assert Double.isFinite(delta);
        biases[binary_string] += delta;
        assert Double.isFinite(biases[binary_string]);

        for (int weight_idx = 0; weight_idx < weights[binary_string].length; weight_idx++) {
            weights[binary_string][weight_idx] += delta * forward.get(weight_idx).strength;
        }
    }

    private double getErrorDerivative()
    {
        double error_derivative = 0;
        if(hasRecentBackwardsSignal)
        {
            error_derivative += mergedForwardStrength - recentBackwardsSignal;
        }
        return error_derivative;
    }

    public double[] getWeights(int bitStr) {
        return weights[bitStr].clone(); // A shallow clone is okay here
    }

    public double[][] getWeights() {
        double[][] clone = new double[weights.length][];
        for (int i = 0; i < clone.length; i++) {
            clone[i] = weights[i].clone();
        }
        return clone;
    }

    public double[] getBiases() {
        return biases.clone();
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
    public boolean[] getConnectivity() {
        boolean[] connectivity = new boolean[ID_COUNTER];
        for (Arc arc : outgoing) {
            connectivity[arc.getSendingID()] = true;
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
                .mapToInt(node -> orderedIDMap.get(node.getID()))
                .reduce(0, (result, id_bit) -> result |= id_bit); // effectively the same as a sum in this case
    }

    /**
     * Create an arraylist of arcs from a binary string representation
     * 
     * @param binStr
     * @return
     */
    public ArrayList<Arc> binStrToArcList(int binStr) {
        ArrayList<Arc> arcs = new ArrayList<Arc>(incoming.size());
        for (int i = 0; i < incoming.size(); i++) {
            if ((binStr & 0b1) == 1) {
                arcs.add(incoming.get(i));
            }
            binStr = binStr >> 1;
        }
        return arcs;
    }

    /**
     * Compute the merged signal strength of a set of incoming signals
     * 
     * @param incomingSignals
     * @return
     */
    @Override
    protected double computeMergedSignalStrength(List<Signal> incomingSignals) {

        double[] input_weights = weights[binary_string];

        double strength = IntStream.range(0, input_weights.length)
                .mapToDouble(i -> input_weights[i] * incomingSignals.get(i).strength)
                .sum();

        strength += biases[binary_string];

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

        // Compute the binary string of the incoming signals
        binary_string = nodeSetToBinStr(incomingSignals.stream().map(Signal::getSendingNode).toList());

        // sorting by id to ensure that the weights are applied to the correct
        // node/signal
        incomingSignals.sort((s1, s2) -> Integer.compare(s1.recievingNode.getID(), s2.recievingNode.getID()));

        mergedForwardStrength = computeMergedSignalStrength(incomingSignals);
        assert Double.isFinite(mergedForwardStrength);
        activatedStrength = activationFunction.activator(mergedForwardStrength);
    }

    /**
     * Attempt to send inference signals to all outgoing connections
     */
    @Override
    public void sendInferenceSignals() {
        for (Arc connection : outgoing) {
            // roll and send a signal if successful
            if (connection.rollFilter(mergedForwardStrength)) {
                connection.sendInferenceSignal(activatedStrength);
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
        if (hasRecentBackwardsSignal && !forward.isEmpty())
        {
            hasRecentBackwardsSignal = false;
            updateWeightsAndBias(getErrorDerivative());
        }

        // reinforce backward signals
        if (!backward.isEmpty()) {
            ArrayList<Arc> arcs = binStrToArcList(backwardsBinStr);
            arcs.forEach(arc -> arc.probDist.applyAdjustments());
        }
    }

    /**
     * Send forward signals and record differences of expectation for training
     * 
     * @param
     */
    public void sendForwardSignals() {
        //int count = 0;
        //double[] expectedValues = new double[outgoing.size()];
        for (Arc connection : outgoing) {
            if (connection.rollFilter(mergedForwardStrength)) {
                connection.sendForwardSignal(activatedStrength);
                //expectedValues[count++] = connection.probDist.getMean();
            }
        }

        //error = errorDerivativeOfOutput(expectedValues, count);
    }

    private double errorDerivativeOfOutput(double[] expectedValues, int count) {
        double error = 0;
        for (int i = 0; i < count; i++) {
            error += networkData.errorFunc.error_derivative(mergedForwardStrength, expectedValues[i]);
        }
        return error / count;
    }

    /**
     * Send backwards signals and record differences of expectation for training
     * 
     * @param
     */
    public void sendBackwardsSignals() {
        // Select the backward signal combination
        Convolution[] convolutions = getReverseOutcomes();
        double[] densityWeights = evaluateConvolutions(convolutions);
        backwardsBinStr = selectReverseOutcome(densityWeights) + 1;

        // Sample signal strengths from the selected distribution
        double[] sample = convolutions[backwardsBinStr - 1].sample(recentBackwardsSignal - biases[backwardsBinStr]);

        // Send the backward signals
        ArrayList<Arc> arcs = binStrToArcList(backwardsBinStr);
        for (int i = 0; i < sample.length; i++) {
            ReferenceArc arc_i = (ReferenceArc) arcs.get(i);
            double sample_i = sample[i];
            double sample_inverse = arc_i.recieving.activationFunction.inverse(sample_i);
            arc_i.sendBackwardSignal(sample_inverse); // send signal backwards
            arc_i.probDist.prepareReinforcement(sample_inverse); // prepare to reinforce the distribution
        }
    }

    public Convolution[] getReverseOutcomes() {
        // Loop over all possible incoming signal combinations and record the value of
        // their convolution
        int n_choices = 0b1 << incoming.size();
        Convolution[] convolutions = new Convolution[n_choices - 1];
        for (int binStr = 1; binStr < n_choices; binStr++) {
            // get the arcs corresponding to this bit string
            ArrayList<Arc> arcs = binStrToArcList(binStr);

            // Seperate the arcs into their distributions and activation functions
            ArrayList<FilterDistribution> distributions = arcs.stream()
                    .map(arc -> arc.probDist)
                    .collect(Collectors.toCollection(ArrayList<FilterDistribution>::new));

            ArrayList<ActivationFunction> activators = arcs.stream()
                    .map(arc -> arc.getSendingNode().activationFunction)
                    .collect(Collectors.toCollection(ArrayList<ActivationFunction>::new));

            // Get the weights of the corresponding arcs
            double[] weights = getWeights(binStr);

            // Get the probability density
            convolutions[binStr - 1] = new Convolution(distributions, activators, weights);
        }

        return convolutions;
    }

    public double[] evaluateConvolutions(Convolution[] convolutions) {
        double[] densityWeights = new double[convolutions.length];
        for (int binStr = 1; binStr <= convolutions.length; binStr++) {
            // Shift the signal strength by the bias
            double shiftedStrength = recentBackwardsSignal - biases[binStr];

            densityWeights[binStr - 1] = convolutions[binStr - 1].convolve(shiftedStrength);
            assert Double.isFinite(densityWeights[binStr - 1]);
            assert densityWeights[binStr - 1] >= 0;
        }
        return densityWeights;
    }

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
    public int[] getIncomingConnectionIDs() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getIncomingConnectionIDs'");
    }

    @Override
    public int[] getOutgoingConnectionIDs() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getOutgoingConnectionIDs'");
    }

    @Override
    public void removeAllReferencesTo(int node_id) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'removeAllReferencesTo'");
    }

    @Override
    public boolean removeIncomingConnectionFrom(int node_id) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'removeIncomingConnectionTo'");
    }

    @Override
    public boolean removeOutgoingConnectionTo(int node_id) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'removeOutgoingConnectionTo'");
    }

}
