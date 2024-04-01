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
public class Node implements Comparable<Node> {

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
    private ArrayList<Signal> inference, inferenceNext;

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

    protected ArrayList<Outcome> outcomes;

    /**
     * 
     */
    private HashMap<TimeKey, Double> error_signals;

    private int[] delta_counts;
    private double[] bias_delta;
    private double[][] weights_delta;

    /**
     * The binary string representation of the incoming arcs being sent a backwards
     * signal
     */
    private int backwardsBinStr;

    protected boolean hasValidForwardSignal;

    public Node(final GraphNetwork network, final ActivationFunction activationFunction) {
        id = ID_COUNTER++;
        name = "Node " + id;
        this.network = Objects.requireNonNull(network);
        this.activationFunction = activationFunction;
        incoming = new ArrayList<Arc>();
        outgoing = new ArrayList<Arc>();
        orderedIDMap = new HashMap<>();
        weights = new double[1][1];
        biases = new double[1];
        weights[0] = new double[0];
        delta_counts = new int[1];
        bias_delta = new double[1];
        weights_delta = new double[1][1];
        weights_delta[0] = new double[0];
        error_signals = new HashMap<>();

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

    public Optional<Arc> getOutgoingConnectionTo(Node recievingNode) {
        return outgoing.stream().filter(arc -> arc.recieving == recievingNode).findAny();
    }

    /**
     * Notify this node of a new incoming forward signal
     * 
     * @param signal
     */
    void recieveForwardSignal(Signal signal) {
        appendForward(signal);
        network.notifyNodeActivation(this);
    }

    public void recieveError(int timestep, int key, double error) {
        TimeKey tk = new TimeKey(timestep, key);
        Double error_rate = error_signals.get(tk);
        if (error_rate == null) {
            error_rate = Double.valueOf(0);
        }
        error_rate += error;
        error_signals.put(tk, error_rate);
    }

    public Double getError(int timestep, int key) {
        return error_signals.get(new TimeKey(timestep, key));
    }

    private void appendForward(Signal signal) {
        int signal_id = orderedIDMap.get(signal.sendingNode.id);
        ArrayList<Signal> signals = forwardNext.get(signal_id);
        if (signals == null) {
            signals = new ArrayList<Signal>(1);
            signals.add(signal);
            forwardNext.put(signal_id, signals);
        } else {
            signals.add(signal);
        }
        uniqueIncomingNodeIDs.add(signal.sendingNode.id);
    }

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

        delta_counts = new int[new_size];
        bias_delta = new double[new_size];
        weights_delta = new double[new_size][];

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

        for (int i = 0; i < new_size; i++) {
            weights_delta[i] = new double[weights.length];
        }
    }

    /*
     * protected void updateWeightsAndBias(double error_derivative) {
     * 
     * // compute delta to update the weights and bias
     * double delta = -networkData.getEpsilon() * error_derivative;
     * assert Double.isFinite(delta);
     * biases[binary_string] += delta;
     * 
     * for (int weight_idx = 0; weight_idx < weights[binary_string].length;
     * weight_idx++) {
     * weights[binary_string][weight_idx] += delta *
     * forward.get(weight_idx).strength;
     * }
     * }
     */

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
        signalSet = signalSet.stream().sorted((s1, s2) -> Integer.compare(s1.sendingNode.id, s2.sendingNode.id))
                .toList();
        List<Node> nodeSet = signalSet.stream().map(signal -> signal.sendingNode).toList();
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
    protected double computeMergedSignalStrength(Collection<Signal> incomingSignals, int binary_string) {

        ArrayList<Signal> sortedSignals = new ArrayList<>(incomingSignals);
        // sorting by id to ensure that the weights are applied to the correct
        // node/signal
        sortedSignals.sort((s1, s2) -> Integer.compare(s1.recievingNode.id, s2.recievingNode.id));

        double[] input_weights = weights[binary_string];

        double strength = IntStream.range(0, input_weights.length)
                .mapToDouble(i -> input_weights[i] * sortedSignals.get(i).strength)
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
    public void applyTrainingChanges() {

        // update weights and biases to reinforce forward signals
        // if the error == NAN then this node failed to send a signal to the next
        // if (!forward.isEmpty() && !Double.isNaN(error))
        // updateWeightsAndBias(error);

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
            double shiftedStrength = activatedValue - biases[binStr];

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
        for(Outcome outcome : outcomesAtTime){
            normalization_const += outcome.probability;
        }

        for (Outcome outcome : outcomesAtTime) {
            Double error = error_signals.get(new TimeKey(timestep, outcome.binary_string));
            if (error != null) {
                sendErrorsBackwards(outcome, timestep, error * outcome.probability / normalization_const);
                sendBackwardsSample(outcome, error);
            }
        }
    }

    private void sendErrorsBackwards(Outcome outcomeAtTime, int timestep, double error) {
        int binary_string = outcomeAtTime.binary_string;
        double error_derivative = activationFunction.derivative(outcomeAtTime.netValue) * error;
        double[] weightsOfNodes = weights[binary_string];
        double[] weightsDeltaOfNodes = weights_delta[binary_string];

        // may need to correct for probability weighting
        bias_delta[binary_string] += error_derivative;
        delta_counts[binary_string] += 1;
        for (int i = 0; i < weightsOfNodes.length; i++) {
            weightsDeltaOfNodes[i] += outcomeAtTime.sourceOutputs[i] * error_derivative;

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
        double[] sample = convolutions[backwardsBinStr - 1].sample(activatedValue - biases[backwardsBinStr]);

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
        return name + ": " + outcomes.stream().sorted((o1, o2) -> -Double.compare(o1.probability, o2.probability))
                .toList().toString();
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

    private class TimeKey {
        private final int timestep;
        private final int key;

        public TimeKey(int timestep, int key) {
            this.timestep = timestep;
            this.key = key;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof TimeKey))
                return false;

            TimeKey tk = (TimeKey) o;
            return key == tk.key & timestep == tk.timestep;
        }

        @Override
        public int hashCode() {
            return timestep << 16 + key;
        }
    }

}
