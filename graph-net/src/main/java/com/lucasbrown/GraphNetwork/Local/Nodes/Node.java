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
import com.lucasbrown.GraphNetwork.Local.Edge;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Signal;
import com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators.IProbabilityCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.IValueCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.AdditiveValueCombinator;
import com.lucasbrown.HelperClasses.IterableTools;
import com.lucasbrown.HelperClasses.Structs.Pair;

public class Node implements INode{

    // TODO: remove hard-coded value
    private static double ZERO_THRESHOLD = 1E-12;
    private static final int CATASTROPHE_LIMIT = 10;

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
    protected final ArrayList<Edge> incoming, outgoing;

    private int numInputCombinations;

    /**
     * Forward-training signals
     */
    protected HashMap<Integer, ArrayList<Signal>> forward, forwardNext;

    protected HashSet<Integer> uniqueIncomingNodeIDs;

    /**
     * Maps all incoming node ID's to an int from 0 to the number of incoming nodes
     * -1
     */
    private final HashMap<Integer, Integer> orderedIDMap;

    protected ArrayList<Outcome> outcomes;

    protected boolean hasValidForwardSignal;
    
    private final IValueCombinator valueCombinator;
    private final IProbabilityCombinator probabilityCombinator;

    public Node(GraphNetwork network, final ActivationFunction activationFunction, final IValueCombinator valueCombinator, final IProbabilityCombinator probabilityCombinator) {
        id = ID_COUNTER++;
        name = "INode " + id;
        this.network = Objects.requireNonNull(network);
        this.activationFunction = Objects.requireNonNull(activationFunction);
        this.valueCombinator = Objects.requireNonNull(valueCombinator);
        this.probabilityCombinator = probabilityCombinator;
        incoming = new ArrayList<Edge>();
        outgoing = new ArrayList<Edge>();
        orderedIDMap = new HashMap<>();
        numInputCombinations = 1;

        uniqueIncomingNodeIDs = new HashSet<>();
        outcomes = new ArrayList<>();
        forward = new HashMap<>();
        forwardNext = new HashMap<>();
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
    public IValueCombinator getValueCombinator(){
        return valueCombinator;
    }

    @Override
    public IProbabilityCombinator getProbabilityCombinator(){
        return probabilityCombinator;
    }

    protected int getIndexOfIncomingNode(INode incoming) {
        return orderedIDMap.get(incoming.getID());
    }

    public int getNumInputCombinations() {
        return numInputCombinations;
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
    public boolean addIncomingConnection(Edge connection) {
        valueCombinator.notifyNewIncomingConnection();
        probabilityCombinator.notifyNewIncomingConnection();
        orderedIDMap.put(connection.getSendingID(), numInputCombinations);
        numInputCombinations *= 2;
        return incoming.add(connection);
    }

    @Override
    public ArrayList<Edge> getAllIncomingConnections() {
        return new ArrayList<>(incoming);
    }

    /**
     * Add an outgoing connection to the node
     * 
     * @param connection
     * @return true
     */
    @Override
    public boolean addOutgoingConnection(Edge connection) {
        return outgoing.add(connection);
    }

    @Override
    public ArrayList<Edge> getAllOutgoingConnections() {
        return new ArrayList<>(outgoing);
    }

    @Override
    public Optional<Edge> getOutgoingConnectionTo(INode recievingNode) {
        return outgoing.stream().filter(arc -> arc.recieving.equals(recievingNode)).findAny();
    }

    @Override
    public Optional<Edge> getIncomingConnectionFrom(INode sendingNode) {
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
    @Override
    public ArrayList<Edge> binStrToArcList(int binStr) {
        ArrayList<Edge> arcs = new ArrayList<Edge>(incoming.size());
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
    @Override
    public void acceptSignals() throws InvalidAlgorithmParameterException {
        if (forwardNext.isEmpty()) {
            throw new InvalidAlgorithmParameterException(
                    "handleIncomingSignals should never be called if no signals have been recieved.");
        }

        hasValidForwardSignal = true;
        forward = forwardNext;
        forwardNext = new HashMap<>();
        combinePossibilities();
    }


    /**
     * Create ALL the possible combinations of outcomes for the incoming signals
     */
    private void combinePossibilities() {

        ArrayList<Pair<ArrayList<Signal>, ArrayList<Signal>>> signalPowerSet = IterableTools
                .flatCartesianPowerProductPair(forward.values());

        
        signalPowerSet.stream().parallel().forEach(pair -> 
            {
                pair.u = sortSignalByID(pair.u);
                pair.v = sortSignalByID(pair.v);
            });

        outcomes = signalPowerSet.stream()
                .parallel()
                .filter(set -> !set.u.isEmpty()) // remove the null set
                .map(this::signalSetToOutcome)
                // .filter(outcome -> outcome.probability > ZERO_THRESHOLD)
                .sorted(Outcome::descendingProbabilitiesComparator)
                .limit(CATASTROPHE_LIMIT)
                .collect(Collectors.toCollection(ArrayList::new));

        // assertion to make sure all references to previous nodes are maintained
        // for(Outcome outcome: outcomes){
        //     if(outcome.sourceOutcomes == null){
        //         continue;
        //     }
        //     for(int i = 0 ; i < outcome.sourceOutcomes.length; i++){
        //         boolean containsReference = false;
        //         List<Outcome> sourceOutcomes = forward.get(getIndexOfIncomingNode(outcome.sourceNodes[i])).stream().map(signal -> signal.sourceOutcome).toList();
        //         for(Outcome so : sourceOutcomes){
        //             containsReference |= outcome.sourceOutcomes[i] == so;
        //         }
        //         assert containsReference;
        //     }
        // }


        // assert outcomes.stream().mapToDouble(outcome -> outcome.probability).sum() <= 1
        //         : "Sum of all outcome probabilities must be equal to or less than 1. \nProbability sum = "
        //                 + outcomes.stream().mapToDouble(outcome -> outcome.probability).sum();

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
        Collection<Signal> signalSet = setPair.u;
        List<INode> nodeSet = signalSet.stream().map(signal -> signal.sendingNode).toList();
        outcome.node = this;
        outcome.binary_string = nodeSetToBinStr(nodeSet);
        outcome.sourceTransferProbabilities = getTransferProbabilities(setPair.v);
        outcome.netValue = valueCombinator.computeMergedSignalStrength(signalSet, outcome.binary_string);
        outcome.activatedValue = activationFunction.activator(outcome.netValue);
        outcome.probability = getProbabilityOfSignalSet(signalSet, setPair.v, outcome.binary_string, outcome.sourceTransferProbabilities);
        outcome.sourceKeys = signalSet.stream().mapToInt(Signal::getSourceKey).toArray();
        outcome.sourceOutcomes = outcomesFromSignal(signalSet);

        outcome.root_bin_str = nodeSetToBinStr(setPair.v.stream().map(signal -> signal.sendingNode).toList());
        outcome.allRootOutcomes = outcomesFromSignal(setPair.v);
        return outcome;
    }

    private double[] getTransferProbabilities(Collection<Signal> signals) {
        signals = sortSignalByID(signals);
        int key = nodeSetToBinStr(signals.stream().map(Signal::getSendingNode).toList());
        return probabilityCombinator.getTransferProbabilities(signals, key);
    }


    public double getProbabilityOfSignalSet(Collection<Signal> signalSet, Collection<Signal> allSendingSignals, int binstr, double[] transferProbs) {
        double probability = 1;
        int i = 0;
        for (Signal s : allSendingSignals) {
            probability *= s.getSourceProbability();
            double transProb = transferProbs[i++];
            if (signalSet.contains(s)) {
                probability *= transProb;
            } else {
                probability *= 1 - transProb;
            }
        }
        return probability;
    }

    private static Outcome[] outcomesFromSignal(Collection<Signal> signals){
        return signals.stream().map(signal -> signal.sourceOutcome).toArray(Outcome[]::new);
    }

    public ArrayList<Signal> sortSignalByID(Collection<Signal> toSort) {
        ArrayList<Signal> sortedSignals = new ArrayList<>(toSort);
        sortedSignals.sort(this::mappedIDComparator);
        return sortedSignals;
    }

    /**
     * Send forward signals and record differences of expectation for training
     * 
     * @param
     */
    @Override
    public void sendForwardSignals() {
        for (Outcome out : outcomes) {
            for (Edge connection : outgoing) {
                connection.sendForwardSignal(out); // oh god
            }
        }
        hasValidForwardSignal = false;
    }

    @Override
    public ArrayList<Outcome> getState() {
        return outcomes;
    }

    private int mappedIDComparator(Signal s1, Signal s2) {
        int i1 = orderedIDMap.get(s1.sendingNode.getID());
        int i2 = orderedIDMap.get(s2.sendingNode.getID());
        return i1 - i2;
    }

    @Override
    public void clearSignals() {
        hasValidForwardSignal = false;
        forward.clear();
        forwardNext.clear();
    }

    @Override
    public String toString() {
        return name + ": " + outcomes.stream()
                // .filter(outcome -> outcome.probability > 0)
                .sorted(Outcome::descendingProbabilitiesComparator)
                .limit(2)
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

    public static int CompareNodes(INode n1, INode n2) {
        return n1.getID() - n2.getID();
    }

    public static boolean areNodesEqual(INode n1, INode n2) {
        return n1.getID() == n2.getID();
    }
}
