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
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;

/**
 * A neural network using a probabalistic directed graph representation.
 * Training currently a work in progress
 * 
 * Representation allows for both positive and negative reinforcement.
 * Only postive reinforcement is implemented currently.
 */
public abstract class GraphNetwork<T extends Node & IInputNode, E extends Node & IOutputNode> {

    /**
     * An object to encapsulate all network hyperparameters
     */
    protected final SharedNetworkData networkData;

    /**
     * A list of all nodes within the graph network
     */
    protected final ArrayList<Node> nodes;

    /**
     * Maps input node-id's to their corresponding node
     */
    protected final HashMap<Integer, T> input_nodes;

    /**
     * Maps output node-id's to their corresponding node
     */
    protected final HashMap<Integer, E> output_nodes;

    /**
     * A hash set containing every node that recieved a signal this step
     */
    protected HashSet<Node> activeNodes;

    /**
     * A hash set containing every node that will recieve a signal in the next step
     */
    protected HashSet<Node> activeNextNodes;

    /**
     * An operation which is to be defined by the user to set the values of input
     * nodes
     */
    private Consumer<HashMap<Integer, T>> inputOperation;

    /**
     * An operation which is to be defined by the user to correct the values of
     * output nodes during training or get output data
     */
    private Consumer<HashMap<Integer, E>> outputOperation;

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

    /**
     * Set the input operation for the network.
     * The input operation accepts a hashmap of id-Node pairs corresponding to input nodes.
     * The operation is expected to act on input nodes according to the user's desires.
     * 
     * @param inputOperation
     */
    public void setInputOperation(Consumer<HashMap<Integer, T>> inputOperation) {
        this.inputOperation = inputOperation == null ? (_1) -> {
        } : inputOperation;
    }

    /**
     * Set the output operation for the network.
     * The output operation accepts a hashmap of id-Node pairs corresponding to output nodes.
     * The operation is expected to interpret the results of the network.
     * 
     * @param outputOperation
     */
    public void setOutputOperation(Consumer<HashMap<Integer, E>> outputOperation) {
        this.outputOperation = outputOperation == null ? (_1) -> {
        } : outputOperation;
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
        activeNodes.forEach(Node::sendInferenceSignals);
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

    public void deactivateAll() {
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
        Node node = getNewHiddenNode(activationFunction);
        nodes.add(node);
        return node;
    }

    public final T createInputNode(final ActivationFunction activationFunction)
    {
        T node = getNewInputNode(activationFunction);
        nodes.add(node);
        input_nodes.put(node.getID(), node);
        return node;
    }

    public final E createOutputNode(final ActivationFunction activationFunction)
    {
        E node = getNewOutputNode(activationFunction);
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

    /**
     * Creates a new hidden node and add it to the network
     * 
     * @return The node that was created
     */
    protected abstract Node getNewHiddenNode(final ActivationFunction activationFunction);

    /**
     * Creates a new input node and add it to the network
     * 
     * @return The node that was created
     */
    protected abstract T getNewInputNode(final ActivationFunction activationFunction);

    /**
     * Creates a new output node and add it to the network
     * 
     * @return The node that was created
     */
    protected abstract E getNewOutputNode(final ActivationFunction activationFunction);

    /**
     * Create a directed edge from the transmitting node to the recieving node.
     * 
     * @param transmittingNode
     * @param recievingNode
     * @param transferFunction
     */
    public abstract void addNewConnection(Node transmittingNode, Node recievingNode,
            FilterDistribution transferFunction);

}
