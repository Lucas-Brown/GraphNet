package com.lucasbrown.GraphNetwork.Global;

import java.security.InvalidAlgorithmParameterException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.TreeSet;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.FilterDistribution;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;

/**
 * A neural network using a probabalistic directed graph representation.
 * Training currently a work in progress
 * 
 * Representation allows for both positive and negative reinforcement.
 * Only postive reinforcement is implemented currently.
 */
public abstract class GraphNetwork {

    /**
     * An object to encapsulate all network hyperparameters
     */
    protected final SharedNetworkData networkData;

    /**
     * A list of all nodes within the graph network
     */
    protected final ArrayList<Node> nodes;

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
    private Runnable inputOperation;

    /**
     * An operation which is to be defined by the user to correct the values of
     * output nodes during training or get output data
     */
    private Runnable outputOperation;

    public GraphNetwork() {
        // TODO: remove hardcoding
        networkData = new SharedNetworkData(new ErrorFunction.MeanSquaredError(), 0.1);

        nodes = new ArrayList<>();
        activeNodes = new HashSet<>();
        activeNextNodes = new HashSet<>();
        inputOperation = () -> {
        };
        outputOperation = () -> {
        };
    }

    /**
     * Set the input operation for the network.
     * input operations are expected to act on input nodes and may depend on
     * external factors such as a reference time.
     * 
     * @param inputOperation
     */
    public void setInputOperation(Runnable inputOperation) {
        this.inputOperation = inputOperation == null ? () -> {
        } : inputOperation;
    }

    /**
     * Set the output operation for the network.
     * output operations are expected to interpret the data exposed by output nodes.
     * 
     * @param outputOperation
     */
    public void setOutputOperation(Runnable outputOperation) {
        this.outputOperation = outputOperation == null ? () -> {
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
        inputOperation.run();
        recieveSignals();
        outputOperation.run();
        sendInferenceSignals();
    }

    /**
     * 
     */
    public void trainingStep() {
        inputOperation.run();
        outputOperation.run();
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

    public String allActiveNodesString() {
        TreeSet<Node> sSet = new TreeSet<Node>(activeNodes);
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
        activeNodes = new HashSet<>();
        activeNextNodes = new HashSet<>();
    }

    public static <T> boolean hasIntersection(HashSet<T> s1, HashSet<T> s2) {
        HashSet<T> intersection = new HashSet<T>(s1);
        intersection.retainAll(s2);
        return intersection.size() > 0;
    }

    /**
     * Creates a new hidden node and add it to the network
     * 
     * @return The node that was created
     */
    public abstract Node createHiddenNode(final ActivationFunction activationFunction);

    /**
     * Creates a new input node and add it to the network
     * 
     * @return The node that was created
     */
    public abstract Node createInputNode(final ActivationFunction activationFunction);

    /**
     * Creates a new output node and add it to the network
     * 
     * @return The node that was created
     */
    public abstract Node createOutputNode(final ActivationFunction activationFunction);

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
