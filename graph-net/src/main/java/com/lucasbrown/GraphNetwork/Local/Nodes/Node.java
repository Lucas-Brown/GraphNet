package com.lucasbrown.GraphNetwork.Local;

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

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of
 * sending a signal out to other connected nodes
 */
public abstract class Node implements Comparable<Node> {

    private final Random rng = new Random();

    /**
     * The coutner is used to give each node a unique ID
     */
    private static int ID_COUNTER = 0;

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
    public final GraphNetwork network;

    /**
     * The activation function
     */
    public final ActivationFunction activationFunction;

    /**
     * All incoming and outgoing node connections.
     */
    protected final ArrayList<Arc> incoming, outgoing;

    /**
     * Forward-training signals
     */
    protected HashMap<Integer, ArrayList<Signal>> forward, forwardNext;

    protected HashSet<Integer> uniqueIncomingNodeIDs;

    /**
     * backward training signals
     */
    protected ArrayList<Signal> backward, backwardNext;

    protected boolean hasRecentBackwardsSignal;

    /**
     * The most recent backwards signal to be compared to by forward signals
     */
    protected ArrayList<Signal> inference, inferenceNext;

    /**
     * Maps all incoming node ID's to an int from 0 to the number of incoming nodes
     * -1
     */
    protected final HashMap<Integer, Integer> orderedIDMap;

    protected ArrayList<Outcome> outcomes;

    protected boolean hasValidForwardSignal;

    public Node(final GraphNetwork network, final ActivationFunction activationFunction) {
        id = ID_COUNTER++;
        name = "Node " + id;
        this.network = Objects.requireNonNull(network);
        this.activationFunction = activationFunction;
        incoming = new ArrayList<Arc>();
        outgoing = new ArrayList<Arc>();

        uniqueIncomingNodeIDs = new HashSet<>();
        outcomes = new ArrayList<>();
        inference = new ArrayList<>();
        inferenceNext = new ArrayList<>();
        forward = new HashMap<>();
        forwardNext = new HashMap<>();
        backward = new ArrayList<>();
        backwardNext = new ArrayList<>();
        hasRecentBackwardsSignal = false;
    }

    public int getID() {
        return id;
    }

    public void setName(String name) {
        this.name = name;
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

    public Optional<Arc> getOutgoingConnectionTo(Node recievingNode) {
        return outgoing.stream().filter(arc -> arc.recieving == recievingNode).findAny();
    }

    /**
     * Notify this node of a new incoming forward signal
     * 
     * @param signal
     */
    void recieveForwardSignal(Signal signal) {
        network.notifyNodeActivation(this);
    }

    public abstract void recieveError(int timestep, int key, double error);

    public abstract Double getError(int timestep, int key);

    /**
     * Notify this node of a new incoming backward signal
     * 
     * @param signal
     */
    void recieveBackwardSignal(Signal signal) {
        backwardNext.add(signal);
        network.notifyNodeActivation(this);
    }

    /**
     * Notify this node of a new inference signal
     * 
     * @param signal
     */
    void recieveInferenceSignal(Signal signal) {
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

    public abstract double[] getWeights(int bitStr);

    public abstract double getBias(int bitStr);

    /**
     * map every incoming node id to its corresponding value and combine.
     * for example, an id of 6 may map to 0b0010 and an id of 2 may map to 0b1000
     * binary_string will thus contain the value 0b1010
     * 
     * @param incomingSignals
     * @return a bit string indicating the weights, bias, and error index to use for
     *         the given set of signals
     */
    public int nodeSetToBinStr(Collection<Node> incomingNodes) {
        return incomingNodes.stream()
                .mapToInt(node -> orderedIDMap.get(node.id))
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
            /*
             * inference = inferenceNext;
             * inferenceNext = new ArrayList<Signal>();
             * hasValidForwardSignal = true;
             * acceptIncomingForwardSignals(inference);
             */
        } else {
            if (!forwardNext.isEmpty()) {
                hasValidForwardSignal = true;
                forward = forwardNext;
                forwardNext = new HashMap<>();
                combinePossibilities();
            }
            if (!backwardNext.isEmpty()) {
                /*
                 * hasRecentBackwardsSignal = true;
                 * backward = backwardNext;
                 * backwardNext = new ArrayList<Signal>();
                 * recentBackwardsSignal = getRecentBackwardsSignal();
                 */
            }
        }

    }

    /**
     * Create ALL the possible combinations of outcomes for the incoming signals
     */
    private void combinePossibilities() {
        HashSet<HashSet<Signal>> signalPowerSet = ArrayTools.flatCartesianPowerProduct(forward.values());
        outcomes = signalPowerSet.stream()
                .filter(set -> !set.isEmpty()) // remove the null set
                .map(this::signalSetToOutcome)
                .collect(Collectors.toCollection(ArrayList::new));
    }

    /**
     * Creates and fills the fields of a new outcome object for a given set of
     * incoming signals
     * 
     * @param signalSet
     * @return
     */
    private Outcome signalSetToOutcome(Collection<Signal> signalSet) {
        Outcome outcome = new Outcome();
        signalSet = signalSet.stream().sorted(Signal::CompareSendingNodeIDs).toList();
        List<Node> nodeSet = signalSet.stream().map(Signal::getSendingNode).toList();
        outcome.binary_string = nodeSetToBinStr(nodeSet);
        outcome.netValue = computeMergedSignalStrength(signalSet, outcome.binary_string);
        outcome.activatedValue = activationFunction.activator(outcome.netValue);
        outcome.probability = getProbabilityOfSignalSet(signalSet);
        outcome.sourceOutputs = signalSet.stream().mapToDouble(Signal::getOutputStrength).toArray();
        outcome.sourceNodes = nodeSet.toArray(new Node[nodeSet.size()]);
        outcome.sourceKeys = signalSet.stream().mapToInt(Signal::getSourceKey).toArray();
        return outcome;
    }

    private double getProbabilityOfSignalSet(Collection<Signal> signalSet) {
        double probability = 1;
        for (Signal s : signalSet) {
            if (uniqueIncomingNodeIDs.contains(s.sendingNode.id)) {
                probability *= s.probability;
            } else {
                probability *= 1 - s.probability;
            }
        }
        return probability;
    }

    private double getRecentBackwardsSignal() {
        return backward.stream().mapToDouble(Signal::getOutputStrength).average().getAsDouble();
    }

    /**
     * Compute the merged signal strength of a set of incoming signals
     * 
     * @param incomingSignals
     * @return
     */
    protected abstract double computeMergedSignalStrength(Collection<Signal> incomingSignals, int binary_string);

    /**
     * Set the state parameters for incoming signal (i.e, either the forward or
     * inference signal)
     * 
     * @param incomingSignals
     */
    /*
     * protected void acceptIncomingForwardSignals(ArrayList<Signal>
     * incomingSignals) {
     * if (incomingSignals.size() == 0)
     * return;
     * hasValidForwardSignal = true;
     * 
     * // Compute the binary string of the incoming signals
     * binary_string =
     * nodeSetToBinStr(incomingSignals.stream().map(Signal::getSendingNode).toList()
     * );
     * 
     * 
     * mergedForwardStrength = computeMergedSignalStrength(incomingSignals);
     * assert Double.isFinite(mergedForwardStrength);
     * activatedStrength = activationFunction.activator(mergedForwardStrength);
     * }
     */

    /**
     * Attempt to send inference signals to all outgoing connections
     */
    /*
     * public void sendInferenceSignals() {
     * for (Arc connection : outgoing) {
     * // roll and send a signal if successful
     * if (connection.rollFilter(mergedForwardStrength)) {
     * connection.sendInferenceSignal(activatedStrength);
     * }
     * }
     * hasValidForwardSignal = false;
     * }
     */

    /**
     * Attempt to send forward and backward signals
     */
    public void sendTrainingSignals() {

        if (!outgoing.isEmpty()) {
            // Send the forward signals and record the cumulative error
            sendForwardSignals();
            hasValidForwardSignal = false;
        }

        /*
         * if (!incoming.isEmpty()) {
         * sendBackwardsSignals();
         * }
         */

    }

    /**
     * Apply all changes. Must proceed a call to trainingStep
     */
    /*
     * public void applyTrainingChanges() {
     * 
     * // update weights and biases to reinforce forward signals
     * // if the error == NAN then this node failed to send a signal to the next
     * // if (!forward.isEmpty() && !Double.isNaN(error))
     * // updateWeightsAndBias(error);
     * 
     * // reinforce backward signals
     * if (!backward.isEmpty()) {
     * ArrayList<Arc> arcs = binStrToArcList(backwardsBinStr);
     * arcs.forEach(arc -> arc.probDist.applyAdjustments());
     * }
     * }
     */

    /**
     * Send forward signals and record differences of expectation for training
     * 
     * @param
     */
    public void sendForwardSignals() {
        for (Outcome out : outcomes) {
            for (Arc connection : outgoing) {
                connection.sendForwardSignal(out.binary_string, out.activatedValue,
                        out.probability * connection.probDist.sendChance(out.netValue)); // oh god
            }
        }

    }

    /*
     * private double errorDerivativeOfOutput(double[] expectedValues, int count) {
     * double error = 0;
     * for (int i = 0; i < count; i++) {
     * error += networkData.errorFunc.error_derivative(mergedForwardStrength,
     * expectedValues[i]);
     * }
     * return error / count;
     * }
     */

    /**
     * Send backwards signals and record differences of expectation for training
     * 
     * @param
     */

    // private void sendBackwardsSignals() {
    // // Select the backward signal combination
    // Convolution[] convolutions = getReverseOutcomes();
    // double[] densityWeights = evaluateConvolutions(convolutions);
    // backwardsBinStr = selectReverseOutcome(densityWeights) + 1;

    // // Sample signal strengths from the selected distribution
    // double[] sample = convolutions[backwardsBinStr -
    // 1].sample(mergedBackwardStrength);

    // // Send the backward signals
    // ArrayList<Arc> arcs = binStrToArcList(backwardsBinStr);
    // for (int i = 0; i < sample.length; i++) {
    // Arc arc_i = arcs.get(i);
    // double sample_i = sample[i];
    // double sample_inverse = arc_i.recieving.activationFunction.inverse(sample_i);
    // arc_i.sendBackwardSignal(sample_inverse); // send signal backwards
    // arc_i.probDist.prepareReinforcement(sample_inverse); // prepare to
    // reinforce// the distribution
    // }

    public FilterDistributionConvolution[] getReverseOutcomes() {
        // Loop over all possible incoming signal combinations and record the value
        // of their convolution
        int n_choices = 0b1 << incoming.size();
        FilterDistributionConvolution[] convolutions = new FilterDistributionConvolution[n_choices - 1];
        for (int binStr = 1; binStr < n_choices; binStr++) {
            // get the arcs corresponding to this bit string
            ArrayList<Arc> arcs = binStrToArcList(binStr);

            // Seperate the arcs into their distributions and activation functions
            ArrayList<FilterDistribution> distributions = arcs.stream()
                    .map(arc -> arc.probDist)
                    .collect(Collectors.toCollection(ArrayList<FilterDistribution>::new));

            ArrayList<ActivationFunction> activators = arcs.stream()
                    .map(arc -> arc.sending.activationFunction)
                    .collect(Collectors.toCollection(ArrayList<ActivationFunction>::new));

            // Get the weights of the corresponding arcs
            double[] weights = getWeights(binStr);

            // Get the probability density
            convolutions[binStr - 1] = new FilterDistributionConvolution(distributions, activators,
                    weights);
        }

        return convolutions;
    }

    public double[] evaluateConvolutions(FilterDistributionConvolution[] convolutions, double activatedValue) {
        double[] densityWeights = new double[convolutions.length];
        for (int binStr = 1; binStr <= convolutions.length; binStr++) {
            // Shift the signal strength by the bias
            double shiftedStrength = activatedValue - getBias(binStr);

            densityWeights[binStr - 1] = convolutions[binStr -
                    1].convolve(shiftedStrength);
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

    public ArrayList<Outcome> getState() {
        return outcomes;
    }

    public void sendErrorsBackwards(ArrayList<Outcome> outcomesAtTime, int timestep) {

        double normalization_const = 0;
        for (Outcome outcome : outcomesAtTime) {
            normalization_const += outcome.probability;
        }

        for (Outcome outcome : outcomesAtTime) {
            Double error = getError(timestep, outcome.binary_string);
            if (error != null) {
                sendErrorsBackwards(outcome, timestep, error * outcome.probability / normalization_const);
                sendBackwardsSample(outcome, error);
            }
        }
    }

    protected abstract void addBiasDelta(int bitStr, double error);

    protected abstract void addWeightDelta(int bitStr, int weight_index, double error);

    private void sendErrorsBackwards(Outcome outcomeAtTime, int timestep, double error) {
        int binary_string = outcomeAtTime.binary_string;
        double error_derivative = activationFunction.derivative(outcomeAtTime.netValue) * error;
        double[] weightsOfNodes = weights[binary_string];

        // may need to correct for probability weighting
        addBiasDelta(binary_string, error_derivative);
        for (int i = 0; i < weightsOfNodes.length; i++) {
            addWeightDelta(binary_string, i, outcomeAtTime.sourceOutputs[i] * error_derivative);

            Node sourceNode = outcomeAtTime.sourceNodes[i];
            sourceNode.recieveError(timestep - 1, outcomeAtTime.sourceKeys[i], error_derivative * weightsOfNodes[i]);

            // apply error as new point for the distribution
            // Arc connection = sourceNode.getOutgoingConnectionTo(this).get();
            // connection.probDist.prepareReinforcement(outcomeAtTime.netValue - error);
        }

    }

    private void sendBackwardsSample(Outcome outcomeAtTime, double error) {
        double activatedValue = outcomeAtTime.netValue - error;

        // Select the backward signal combination
        FilterDistributionConvolution[] convolutions = getReverseOutcomes();
        double[] densityWeights = evaluateConvolutions(convolutions, activatedValue);
        backwardsBinStr = selectReverseOutcome(densityWeights) + 1;

        // Sample signal strengths from the selected distribution
        double[] sample = convolutions[backwardsBinStr - 1].sample(activatedValue - getBias(backwardsBinStr));

        // Send the backward signals
        ArrayList<Arc> arcs = binStrToArcList(backwardsBinStr);
        for (int i = 0; i < sample.length; i++) {
            Arc arc_i = arcs.get(i);
            double sample_i = sample[i];
            double sample_inverse = arc_i.recieving.activationFunction.inverse(sample_i);
            assert Double.isFinite(sample_inverse);
            arc_i.probDist.prepareReinforcement(sample_inverse); // prepare to reinforce// the distribution
        }
    }

    public void applyErrorSignals(double epsilon) {
        for (int key = 1; key < biases.length; key++) {
            int count = delta_counts[key];
            if (count == 0)
                continue;

            double delta = -epsilon / count;
            biases[key] += delta * bias_delta[key];
            bias_delta[key] = 0;

            for (int i = 0; i < weights[key].length; i++) {
                weights[key][i] += delta * weights_delta[key][i];
                weights_delta[key][i] = 0;
            }

            delta_counts[key] = 0;
        }

        for (Arc connection : outgoing) {
            connection.probDist.applyAdjustments();
        }

        error_signals.clear();
    }

    public void clearSignals() {
        hasRecentBackwardsSignal = false;
        hasValidForwardSignal = false;
        inference.clear();
        forward.clear();
        backward.clear();
        inferenceNext.clear();
        forwardNext.clear();
        backwardNext.clear();
    }

    @Override
    public String toString() {
        return name + ": " + outcomes.stream()
                .sorted(Signal::CompareSendingNodeIDs)
                .toList()
                .toString();
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
