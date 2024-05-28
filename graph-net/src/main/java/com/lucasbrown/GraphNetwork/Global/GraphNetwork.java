package com.lucasbrown.GraphNetwork.Global;

import java.security.InvalidAlgorithmParameterException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeSet;
import java.util.Map.Entry;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Nodes.IInputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.IOutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.DataSetTraining.IExpectationAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.IFilter;

/**
 * A neural network using a probabalistic directed graph representation.
 * Training currently a work in progress
 * 
 * Current representation allows for both positive and negative reinforcement.
 * Only postive reinforcement is implemented currently.
 */
public class GraphNetwork {

    // Temporary global variable for testing
    public static final double N_MAX = 1000;


    /**
     * A list of all nodes within the graph network
     */
    private final ArrayList<INode> nodes;

    /**
     * All input nodes
     */
    private HashMap<Integer, InputNode> input_nodes;

    /**
     * All output nodes
     */
    private HashMap<Integer, OutputNode> output_nodes;

    /**
     * A hash set containing every node that recieved a signal this step
     */
    private HashSet<INode> activeNodes;

    /**
     * A hash set containing every node that will recieve a signal in the next step
     */
    private HashSet<INode> activeNextNodes;

    /**
     * An operation which is to be defined by the user to set the values of input
     * nodes
     */
    private Consumer<HashMap<Integer, ? extends IInputNode>> inputOperation;

    /**
     * An operation which is to be defined by the user to correct the values of
     * output nodes during training or get output data
     */
    private Consumer<HashMap<Integer, ? extends IOutputNode>> outputOperation;

    public GraphNetwork() {

        nodes = new ArrayList<>();
        input_nodes = new HashMap<>();
        output_nodes = new HashMap<>();
        activeNodes = new HashSet<>();
        activeNextNodes = new HashSet<>();
        inputOperation = (_1) -> {
        };
        outputOperation = (_1) -> {
        };
    }

    public void setInputOperation(Consumer<HashMap<Integer, ? extends IInputNode>> inputOperation) {
        this.inputOperation = inputOperation == null ? (_1) -> {
        } : inputOperation;
    }

    public void setOutputOperation(Consumer<HashMap<Integer, ? extends IOutputNode>> outputOperation) {
        this.outputOperation = outputOperation == null ? (_1) -> {
        } : outputOperation;
    }

    public ArrayList<OutputNode> getOutputNodes() {
        return getSortedNodes(output_nodes);
    }

    public ArrayList<InputNode> getInputNodes() {
        return getSortedNodes(input_nodes);
    }

    private <T> ArrayList<T> getSortedNodes(HashMap<Integer, T> map) {
        return map.entrySet()
                .stream()
                .sorted(GraphNetwork::IntegerEntryComparator)
                .map(Entry::getValue)
                .collect(Collectors.toCollection(ArrayList::new));
    }

    /**
     * Creates a new node and adds it to the list of nodes within the network
     * TODO: Potentially remove external addition of nodes and connections in favor
     * of dynamically adding nodes/edges during training
     * 
     * @return The node that was created
     */
    /*
     * public INode createHiddenNode(final ActivationFunction activationFunction) {
     * INode n = new INode(this, networkData, activationFunction);
     * nodes.add(n);
     * return n;
     * }
     * 
     * public InputNode createInputNode(final ActivationFunction activationFunction)
     * {
     * InputNode n = new InputNode(this, networkData, activationFunction);
     * nodes.add(n);
     * return n;
     * }
     * 
     * public OutputNode createOutputNode(final ActivationFunction
     * activationFunction) {
     * OutputNode n = new OutputNode(this, networkData, activationFunction);
     * nodes.add(n);
     * return n;
     * }
     */

    public Arc addNewConnection(INode transmittingNode, INode recievingNode,
            IFilter transferFunction, IExpectationAdjuster filterAdjuster) {
        
        Arc connection = new Arc(transmittingNode, recievingNode, transferFunction, filterAdjuster);
        transmittingNode.addOutgoingConnection(connection);
        recievingNode.addIncomingConnection(connection);
        return connection;
    }

    /**
     * notify the network that a node has been activated.
     * 
     * @param activatedNode
     */
    public void notifyNodeActivation(INode activatedNode) {
        activeNextNodes.add(activatedNode);
    }

    /**
     * Step the entire network forward one itteration
     * Does not train the network
     */
    public void inferenceStep() {
        inputOperation.accept(input_nodes);
        recieveSignals();
        outputOperation.accept(output_nodes);
        sendInferenceSignals();
    }

    /**
     * 
     */
    public void trainingStep() {
        inputOperation.accept(input_nodes);
        outputOperation.accept(output_nodes);
        recieveSignals();
        sendTrainingSignals();
    }

    public boolean isNetworkDead() {
        return activeNodes.isEmpty();
    }

    /**
     * Tell all active nodes to accept all incoming signals
     */
    private void recieveSignals() {
        activeNodes = activeNextNodes;
        activeNextNodes = new HashSet<>();
        activeNodes.forEach(t -> {
            try {
                t.acceptSignals();
            } catch (InvalidAlgorithmParameterException e) {
                e.printStackTrace();
            }
        });
    }

    private void sendInferenceSignals() {
        // activeNodes.forEach(INode::sendInferenceSignals);
    }

    /**
     * Tell each node to compute next phase of signals
     */
    private void sendTrainingSignals() {
        // Will automatically collect generated signals in {@code activeNextNodes}
        activeNodes.stream().forEach(INode::sendTrainingSignals);
        //activeNodes.stream().forEach(INode::applyTrainingChanges);
    }

    @Override
    public String toString() {
        // List<INode> activeForwardNodes =
        // activeNodes.stream().filter(INode::hasValidForwardSignal).toList();
        return nodesToString(activeNodes);
    }

    public static String nodesToString(Collection<INode> nodes) {
        TreeSet<INode> sSet = new TreeSet<INode>(nodes);
        StringBuilder sb = new StringBuilder();
        sSet.forEach(node -> {
            sb.append(node.toString());
            sb.append('\t');
        });
        return sb.toString();
    }

    public void deactivateAll() {
        activeNodes.forEach(INode::clearSignals);
        activeNextNodes.forEach(INode::clearSignals);
        activeNodes.clear();
        activeNextNodes.clear();
    }

    public void addNodeToNetwork(INode node){
        node.setParentNetwork(this);
        nodes.add(node);
        if(node instanceof InputNode){
            input_nodes.put(node.getID(), (InputNode) node);
        }
        if(node instanceof OutputNode){
            output_nodes.put(node.getID(), (OutputNode) node);
        }
    }

    public INode getNode(int id) {
        return nodes.get(id);
    }

    public ArrayList<INode> getNodes() {
        return new ArrayList<INode>(nodes);
    }

    public ArrayList<INode> getActiveNodes() {
        return new ArrayList<INode>(activeNodes);
    }

    private static int IntegerEntryComparator(Entry<Integer, ?> e1, Entry<Integer, ?> e2) {
        return Integer.compare(e1.getKey(), e2.getKey());
    }
}
