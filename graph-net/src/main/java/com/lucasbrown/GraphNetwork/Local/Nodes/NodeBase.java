package com.lucasbrown.GraphNetwork.Local.Nodes;

import java.security.InvalidAlgorithmParameterException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Random;
import java.util.stream.Collectors;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Signal;
import com.lucasbrown.NetworkTraining.ApproximationTools.ArrayTools;
import com.lucasbrown.NetworkTraining.ApproximationTools.Pair;
import com.lucasbrown.NetworkTraining.ApproximationTools.Convolution.FilterDistributionConvolution;
import com.lucasbrown.NetworkTraining.DataSetTraining.BackwardsSamplingDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.IExpectationAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.ITrainableDistribution;

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of
 * sending a signal out to other connected nodes
 */
public abstract class NodeBase implements INode {

    // TODO: remove hard-coded value
    private static double ZERO_THRESHOLD = 1E-12;

    protected final Random rng = new Random();

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
    protected GraphNetwork network;

    /**
     * The activation function
     */
    public final ActivationFunction activationFunction;

    /**
     * All incoming and outgoing node connections.
     */
    protected final ArrayList<Arc> incoming, outgoing;

    /**
     * The distribution of outputs produced by this node
     */
    protected ITrainableDistribution outputDistribution;

    /**
     * An objects which adjusts the parameters of outputDistribution given new data
     */
    protected IExpectationAdjuster outputAdjuster;

    /**
     * The probability distribution corresponding to signal passes
     */
    public ITrainableDistribution signalChanceDistribution;

    public IExpectationAdjuster chanceAdjuster;

    private int incomingPowerSetSize;

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
    private final HashMap<Integer, Integer> orderedIDMap;

    protected ArrayList<Outcome> outcomes;

    protected boolean hasValidForwardSignal;

    public NodeBase(GraphNetwork network, final ActivationFunction activationFunction,
            ITrainableDistribution outputDistribution,
            ITrainableDistribution signalChanceDistribution) {
        this(network, activationFunction, outputDistribution,
                outputDistribution.getDefaulAdjuster().apply(outputDistribution),
                signalChanceDistribution, signalChanceDistribution.getDefaulAdjuster().apply(signalChanceDistribution));
    }

    public NodeBase(GraphNetwork network, final ActivationFunction activationFunction,
            ITrainableDistribution outputDistribution, IExpectationAdjuster outputAdjuster,
            ITrainableDistribution signalChanceDistribution, IExpectationAdjuster chanceAdjuster) {
        id = ID_COUNTER++;
        name = "INode " + id;
        this.network = network;
        this.activationFunction = activationFunction;
        this.outputDistribution = outputDistribution;
        this.outputAdjuster = outputAdjuster;
        this.signalChanceDistribution = signalChanceDistribution;
        this.chanceAdjuster = chanceAdjuster;
        incoming = new ArrayList<Arc>();
        outgoing = new ArrayList<Arc>();
        orderedIDMap = new HashMap<>();
        incomingPowerSetSize = 1;

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

    @Override
    public int getID() {
        return id;
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public void setName(String name) {
        this.name = name;
    }

    @Override
    public void setParentNetwork(GraphNetwork network) {
        this.network = network;
    }

    @Override
    public GraphNetwork getParentNetwork() {
        return network;
    }

    @Override
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    @Override
    public ITrainableDistribution getOutputDistribution() {
        return outputDistribution;
    }

    @Override
    public ITrainableDistribution getSignalChanceDistribution() {
        return signalChanceDistribution;
    }

    protected int getIndexOfIncomingNode(INode incoming) {
        return orderedIDMap.get(incoming.getID());
    }

    public int getIncomingPowerSetSize() {
        return incomingPowerSetSize;
    }

    /**
     * 
     * @param node
     * @return whether this node is connected to the provided node
     */
    @Override
    public boolean doesContainConnection(INode node) {
        return outgoing.stream().anyMatch(connection -> connection.doesMatchNodes(this, node));
    }

    /**
     * Add an incoming connection to the node
     * 
     * @param connection
     * @return true
     */
    @Override
    public boolean addIncomingConnection(Arc connection) {
        orderedIDMap.put(connection.getSendingID(), incomingPowerSetSize);
        incomingPowerSetSize *= 2;
        return incoming.add(connection);
    }

    @Override
    public ArrayList<Arc> getAllIncomingConnections() {
        return new ArrayList<>(incoming);
    }

    /**
     * Add an outgoing connection to the node
     * 
     * @param connection
     * @return true
     */
    @Override
    public boolean addOutgoingConnection(Arc connection) {
        return outgoing.add(connection);
    }

    @Override
    public ArrayList<Arc> getAllOutgoingConnections() {
        return new ArrayList<>(outgoing);
    }

    @Override
    public Optional<Arc> getOutgoingConnectionTo(INode recievingNode) {
        return outgoing.stream().filter(arc -> arc.recieving.equals(recievingNode)).findAny();
    }

    @Override
    public Optional<Arc> getIncomingConnectionFrom(INode sendingNode) {
        return incoming.stream().filter(arc -> arc.sending.equals(sendingNode)).findAny();
    }

    /**
     * Notify this node of a new incoming forward signal
     * 
     * @param signal
     */
    @Override
    public void recieveForwardSignal(Signal signal) {
        appendForward(signal);
        network.notifyNodeActivation(this);
    }

    /**
     * Notify this node of a new incoming backward signal
     * 
     * @param signal
     */
    @Override
    public void recieveBackwardSignal(Signal signal) {
        backwardNext.add(signal);
        network.notifyNodeActivation(this);
    }

    /**
     * Notify this node of a new inference signal
     * 
     * @param signal
     */
    @Override
    public void recieveInferenceSignal(Signal signal) {
        inferenceNext.add(signal);
        network.notifyNodeActivation(this);
    }

    /**
     * Get whether the current forward signal is set and valid
     * 
     * @return
     */
    @Override
    public boolean hasValidForwardSignal() {
        return hasValidForwardSignal;
    }

    @Override
    public void setValidForwardSignal(boolean state) {
        hasValidForwardSignal = state;
    }

    private void appendForward(Signal signal) {
        int signal_id = orderedIDMap.get(signal.getSendingID());
        ArrayList<Signal> signals = forwardNext.get(signal_id);
        if (signals == null) {
            signals = new ArrayList<Signal>(1);
            signals.add(signal);
            forwardNext.put(signal_id, signals);
        } else {
            signals.add(signal);
        }
        uniqueIncomingNodeIDs.add(signal.getSendingID());
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
    public int nodeSetToBinStr(Collection<INode> incomingNodes) {
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
     * Use the outcomes to prepare weighted adjustments to the outcome distribution
     */
    public void prepareOutputDistributionAdjustments(ArrayList<Outcome> allOutcomes) {
        for (Outcome o : allOutcomes) {
            // weigh the outcomes by their probability of occurring
            double error = o.errorOfOutcome.hasValues() ? o.errorOfOutcome.getAverage() : 0;
            outputAdjuster.prepareAdjustment(o.probability, new double[] { o.activatedValue - error });
        }
    }

    /**
     * Create ALL the possible combinations of outcomes for the incoming signals
     */
    private void combinePossibilities() {
        HashSet<Pair<HashSet<Signal>, HashSet<Signal>>> signalPowerSet = ArrayTools
                .flatCartesianPowerProductPair(forward.values());

        outcomes = signalPowerSet.stream()
                .filter(set -> !set.u.isEmpty()) // remove the null set
                .map(this::signalSetToOutcome)
                .filter(outcome -> outcome.probability > ZERO_THRESHOLD)
                .collect(Collectors.toCollection(ArrayList::new));

        assert outcomes.stream().mapToDouble(outcome -> outcome.probability).sum() <= 1
                : "Sum of all outcome probabilities must be equal to or less than 1. \nProbability sum = "
                        + outcomes.stream().mapToDouble(outcome -> outcome.probability).sum();
        if (outcomes.isEmpty()) {
            System.out.println();
        }
    }

    /**
     * Creates and fills the fields of a new outcome object for a given set of
     * incoming signals
     * 
     * @param signalSet
     * @return
     */
    private Outcome signalSetToOutcome(Pair<? extends Collection<Signal>, ? extends Collection<Signal>> setPair) {
        Outcome outcome = new Outcome();

        // sorting by id to ensure that the weights are applied to the correct
        // node/signal
        ArrayList<Signal> signalSet = sortSignalByID(setPair.u);
        List<INode> nodeSet = signalSet.stream().map(signal -> signal.sendingNode).toList();
        outcome.binary_string = nodeSetToBinStr(nodeSet);
        outcome.netValue = computeMergedSignalStrength(signalSet, outcome.binary_string);
        outcome.activatedValue = activationFunction.activator(outcome.netValue);
        outcome.probability = getProbabilityOfSignalSet(signalSet, setPair.v);
        outcome.sourceNodes = nodeSet.toArray(new INode[nodeSet.size()]);
        outcome.sourceKeys = signalSet.stream().mapToInt(Signal::getSourceKey).toArray();
        outcome.sourceOutcomes = signalSet.stream().map(signal -> signal.sourceOutcome).toArray(Outcome[]::new);
        return outcome;
    }

    public ArrayList<Signal> sortSignalByID(Collection<Signal> toSort) {
        ArrayList<Signal> sortedSignals = new ArrayList<>(toSort);
        sortedSignals.sort(this::mappedIDComparator);
        return sortedSignals;
    }

    private double getProbabilityOfSignalSet(Collection<Signal> signalSet, Collection<Signal> allSendingSignals) {
        double probability = 1;
        for (Signal s : allSendingSignals) {
            probability *= s.getSourceProbability();
            if (signalSet.contains(s)) {
                probability *= s.getFiringProbability();
            } else {
                probability *= 1 - s.getFiringProbability();
            }
        }
        return probability;
    }

    private double getRecentBackwardsSignal() {
        return backward.stream().mapToDouble(Signal::getOutputStrength).average().getAsDouble();
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
    @Override
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
    @Override
    public void sendForwardSignals() {
        for (Outcome out : outcomes) {
            for (Arc connection : outgoing) {
                connection.sendForwardSignal(out); // oh god
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

    /*
     * public FilterDistributionConvolution[] getReverseOutcomes() {
     * // Loop over all possible incoming signal combinations and record the value
     * // of their convolution
     * int n_choices = 0b1 << incoming.size();
     * FilterDistributionConvolution[] convolutions = new
     * FilterDistributionConvolution[n_choices - 1];
     * for (int binStr = 1; binStr < n_choices; binStr++) {
     * // get the arcs corresponding to this bit string
     * ArrayList<Arc> arcs = binStrToArcList(binStr);
     * 
     * // Get the distributions of output values from the incoming nodes
     * ArrayList<BackwardsSamplingDistribution> distributions = arcs.stream()
     * .map(arc -> arc.sending.getOutputDistribution())
     * .collect(Collectors.toCollection(ArrayList<BackwardsSamplingDistribution>::
     * new));
     * 
     * ArrayList<ActivationFunction> activators = arcs.stream()
     * .map(arc -> arc.sending.getActivationFunction())
     * .collect(Collectors.toCollection(ArrayList<ActivationFunction>::new));
     * 
     * // Get the weights of the corresponding arcs
     * double[] weights = getWeights(binStr);
     * 
     * // Get the probability density
     * convolutions[binStr - 1] = new FilterDistributionConvolution(distributions,
     * activators,
     * weights);
     * }
     * 
     * return convolutions;
     * }
     * 
     * public double[] evaluateConvolutions(FilterDistributionConvolution[]
     * convolutions, double activatedValue) {
     * double[] densityWeights = new double[convolutions.length];
     * for (int binStr = 1; binStr <= convolutions.length; binStr++) {
     * // Shift the signal strength by the bias
     * double shiftedStrength = activatedValue - getBias(binStr);
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
    /*
     * public int selectReverseOutcome(double[] densityWeights) {
     * // get the sum of all weights
     * double total = 0;
     * for (double d : densityWeights) {
     * total += d;
     * }
     * 
     * // Roll a number between 0 and the total
     * double roll = rng.nextDouble() * total;
     * 
     * // Select the first index whose partial sum is over the roll
     * double sum = 0;
     * for (int i = 0; i < densityWeights.length; i++) {
     * sum += densityWeights[i];
     * if (sum >= roll) {
     * return i;
     * }
     * }
     * return -1; // This should never happen
     * }
     */

    @Override
    public void applyDistributionUpdate() {
        outputAdjuster.applyAdjustments();
        chanceAdjuster.applyAdjustments();
    }

    @Override
    public void applyFilterUpdate() {
        for (Arc connection : outgoing) {
            if (connection.filterAdjuster != null) {
                connection.filterAdjuster.applyAdjustments();
            }
        }
    }

    public ArrayList<Outcome> getState() {
        return outcomes;
    }

    @Override
    public void sendErrorsBackwards(Outcome outcomeAtTime) {
        int binary_string = outcomeAtTime.binary_string;
        double error_derivative = activationFunction.derivative(outcomeAtTime.netValue)
                * outcomeAtTime.errorOfOutcome.getProdSum();
        double[] weightsOfNodes = getWeights(binary_string);

        for (int i = 0; i < weightsOfNodes.length; i++) {
            if (!outcomeAtTime.passRate.hasValues()) {
                continue;
            }
            Outcome so = outcomeAtTime.sourceOutcomes[i];
            double pass_avg = outcomeAtTime.passRate.getAverage();
            assert Double.isFinite(pass_avg);
            so.passRate.add(pass_avg, outcomeAtTime.probability);
            so.errorOfOutcome.add(error_derivative * weightsOfNodes[i],
                    outcomeAtTime.probability);

            // apply error as new point for the distribution
            // Arc connection =
            // outcomeAtTime.sourceNodes[i].getOutgoingConnectionTo(this).get();
            // connection.probDist.prepareReinforcement(outcomeAtTime.netValue -
            // error_derivative);
        }

    }

    /*
     * private void sendBackwardsSample(Outcome outcomeAtTime) {
     * double activatedValue = outcomeAtTime.netValue -
     * outcomeAtTime.errorOfOutcome.getAverage();
     * 
     * // Select the backward signal combination
     * FilterDistributionConvolution[] convolutions = getReverseOutcomes();
     * double[] densityWeights = evaluateConvolutions(convolutions, activatedValue);
     * int backwardsBinStr = selectReverseOutcome(densityWeights) + 1;
     * 
     * // Sample signal strengths from the selected distribution
     * double[][] sample = convolutions[backwardsBinStr - 1].sample(activatedValue -
     * getBias(backwardsBinStr), 10);
     * 
     * // Send the backward signals
     * ArrayList<Arc> arcs = binStrToArcList(backwardsBinStr);
     * for (double[] sample_n : sample) {
     * for (int i = 0; i < arcs.size(); i++) {
     * Arc arc_i = arcs.get(i);
     * // double sample_inverse =
     * // arc_i.recieving.getActivationFunction().inverse(sample_i);
     * assert Double.isFinite(sample_n[i]);
     * 
     * // TODO: fix backwards filter training
     * // System.out.println(sample_i);
     * 
     * // arc_i.filter.prepareWeightedReinforcement(sample_n[i],
     * 1/sample[i].length);
     * // // prepare to reinforce the distribution
     * // arc_i.probDist.prepareReinforcement(sample_n[i],
     * // Math.min(1/(outcomeAtTime.probability*sample[i].length),
     * // Math.sqrt(arc_i.probDist.getNumberOfPointsInDistribution())));
     * }
     * }
     * }
     */

    @Override
    public void adjustProbabilitiesForOutcome(Outcome outcome) {
        if (!outcome.passRate.hasValues() || outcome.probability == 0) {
            return;
        }
        double pass_rate = outcome.passRate.getAverage();

        // Add another point for the net firing chance distribution
        chanceAdjuster.prepareAdjustment(outcome.probability, new double[] { pass_rate });

        // Reinforce the filter with the pass rate for each point
        for (int i = 0; i < outcome.sourceNodes.length; i++) {
            INode sourceNode = outcome.sourceNodes[i];

            double error_derivative = outcome.errorOfOutcome.getAverage();
            Arc arc = getIncomingConnectionFrom(sourceNode).get(); // should be guaranteed to exist

            // if the error is not defined and the pass rate is 0, then zero error should be
            // expected
            if (Double.isNaN(error_derivative) && pass_rate == 0) {
                error_derivative = 0;
            }
            assert !Double.isNaN(error_derivative);

            if (arc.filterAdjuster != null) {
                double shifted_value = outcome.sourceOutcomes[i].activatedValue - error_derivative;
                double prob = outcome.sourceOutcomes[i].probability;
                arc.filterAdjuster.prepareAdjustment(prob, new double[] { shifted_value, pass_rate,
                        outputDistribution.getProbabilityDensity(shifted_value) });
            }
        }

    }

    public int mappedIDComparator(Signal s1, Signal s2) {
        int i1 = orderedIDMap.get(s1.sendingNode.getID());
        int i2 = orderedIDMap.get(s2.sendingNode.getID());
        return i1 - i2;
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
                // .filter(outcome -> outcome.probability > 0)
                .sorted(Outcome::descendingProbabilitiesComparator)
                .toList()
                .toString();
    }

    @Override
    public int hashCode() {
        return id;
    }

    @Override
    public int compareTo(INode o) {
        return INode.CompareNodes(this, o);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof INode))
            return false;
        return INode.areNodesEqual(this, (INode) o);
    }

    protected abstract double computeMergedSignalStrength(Collection<Signal> incomingSignals, int binary_string);

    public abstract double[] getWeights(int bitStr);

    public abstract double getBias(int bitStr);

}
