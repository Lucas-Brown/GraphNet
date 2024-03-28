package com.lucasbrown.GraphNetwork.Global;

import java.security.InvalidAlgorithmParameterException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.TreeSet;
import java.util.function.Consumer;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.FilterDistribution;
import com.lucasbrown.GraphNetwork.Local.IInputNode;
import com.lucasbrown.GraphNetwork.Local.IOutputNode;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.InputNode;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.GraphNetwork.Local.OutputNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;

/**
 * A neural network using a probabalistic directed graph representation.
 * Training currently a work in progress
 * 
 * Current representation allows for both positive and negative reinforcement.
 * Only postive reinforcement is implemented currently.
 */
public class GraphNetwork {

    /**
     * An object to encapsulate all network hyperparameters
     */
    private final SharedNetworkData networkData;

    /**
     * A list of all nodes within the graph network
     */
    private final ArrayList<Node> nodes;

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
    private HashSet<Node> activeNodes;

    /**
     * A hash set containing every node that will recieve a signal in the next step
     */
    private HashSet<Node> activeNextNodes;

    /**
     * An operation which is to be defined by the user to set the values of input
     * nodes
     */
    private Consumer<HashMap<Integer, InputNode>> inputOperation;

    /**
     * An operation which is to be defined by the user to correct the values of
     * output nodes during training or get output data
     */
    private Consumer<HashMap<Integer, OutputNode>> outputOperation;

    public GraphNetwork(){
        // TODO: remove hardcoding
        networkData = new SharedNetworkData(new ErrorFunction.MeanSquaredError(), 0.01);

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

    public void setInputOperation(Consumer<HashMap<Integer, InputNode>> inputOperation) {
        this.inputOperation = inputOperation == null ? (_1) -> {
        } : inputOperation;
    }

    public void setOutputOperation(Consumer<HashMap<Integer, OutputNode>> outputOperation) {
        this.outputOperation = outputOperation == null ? (_1) -> {
        } : outputOperation;
    }

    /**
     * Creates a new node and adds it to the list of nodes within the network
     * TODO: Potentially remove external addition of nodes and connections in favor
     * of dynamically adding nodes/edges during training
     * 
     * @return The node that was created
     */
    /*
    public Node createHiddenNode(final ActivationFunction activationFunction) {
        Node n = new Node(this, networkData, activationFunction);
        nodes.add(n);
        return n;
    }

    public InputNode createInputNode(final ActivationFunction activationFunction) {
        InputNode n = new InputNode(this, networkData, activationFunction);
        nodes.add(n);
        return n;
    }

    public OutputNode createOutputNode(final ActivationFunction activationFunction) {
        OutputNode n = new OutputNode(this, networkData, activationFunction);
        nodes.add(n);
        return n;
    }
    */

    public void addNewConnection(Node transmittingNode, Node recievingNode,
            FilterDistribution transferFunction) {
        // boolean doesConnectionExist =
        // transmittingNode.DoesContainConnection(recievingNode);
        // if(!doesConnectionExist)
        // {
        Arc connection = new Arc(transmittingNode, recievingNode, transferFunction);
        transmittingNode.addOutgoingConnection(connection);
        recievingNode.addIncomingConnection(connection);
        // }
        // return doesConnectionExist;
    }

    /**
     * notify the network that a node has been activated.
     * 
     * @param activatedNode
     */
    public void notifyNodeActivation(Node activatedNode) {
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
        //activeNodes.forEach(Node::sendInferenceSignals);
    }

    /**
     * Tell each node to compute next phase of signals
     */
    private void sendTrainingSignals() {
        // Will automatically collect generated signals in {@code activeNextNodes}
        activeNodes.stream().forEach(Node::sendTrainingSignals);
        activeNodes.stream().forEach(Node::applyTrainingChanges);
    }

    @Override
    public String toString()
    {
        //List<Node> activeForwardNodes = activeNodes.stream().filter(Node::hasValidForwardSignal).toList();
        return nodesToString(activeNodes);
    }

    public static String nodesToString(Collection<Node> nodes) {
        TreeSet<Node> sSet = new TreeSet<Node>(nodes);
        StringBuilder sb = new StringBuilder();
        sSet.forEach(node -> {
            sb.append(node.toString());
            sb.append('\t');
        });
        return sb.toString();
    }

    public void deactivateAll()
    {
        activeNodes.forEach(Node::clearSignals);
        activeNextNodes.forEach(Node::clearSignals);
        activeNodes.clear();
        activeNextNodes.clear();
    }

    public static <T> boolean hasIntersection(HashSet<T> s1, HashSet<T> s2) {
        HashSet<T> intersection = new HashSet<T>(s1);
        intersection.retainAll(s2);
        return intersection.size() > 0;
    }
    

    public final Node createHiddenNode(final ActivationFunction activationFunction)
    {
        Node node = new Node(this, networkData, activationFunction);
        nodes.add(node);
        return node;
    }

    public final InputNode createInputNode(final ActivationFunction activationFunction)
    {
        InputNode node = new InputNode(this, networkData, activationFunction);
        nodes.add(node);
        input_nodes.put(node.getID(), node);
        return node;
    }

    public final OutputNode createOutputNode(final ActivationFunction activationFunction)
    {
        OutputNode node = new OutputNode(this, networkData, activationFunction);
        nodes.add(node);
        output_nodes.put(node.getID(), node);
        return node;
    }

    public Node getNode(int id) {
        return nodes.get(id);
    }

    public ArrayList<Node> getNodes()
    {
        return new ArrayList<Node>(nodes);
    }

}
