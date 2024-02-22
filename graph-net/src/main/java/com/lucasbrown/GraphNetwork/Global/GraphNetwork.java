package com.lucasbrown.GraphNetwork.Global;

import java.security.InvalidAlgorithmParameterException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.TreeSet;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.ActivationProbabilityDistribution;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.InputNode;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.GraphNetwork.Local.OutputNode;
import com.lucasbrown.NetworkTraining.ErrorFunction;

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

    public void setInputOperation(Runnable inputOperation) {
        this.inputOperation = inputOperation == null ? () -> {
        } : inputOperation;
    }

    public void setOutputOperation(Runnable outputOperation) {
        this.outputOperation = outputOperation == null ? () -> {
        } : outputOperation;
    }

    /**
     * Creates a new node and adds it to the list of nodes within the network
     * TODO: Potentially remove external addition of nodes and connections in favor
     * of dynamically adding nodes/edges during training
     * 
     * @return The node that was created
     */
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

    public void addNewConnection(Node transmittingNode, Node recievingNode,
            ActivationProbabilityDistribution transferFunction) {
        // boolean doesConnectionExist =
        // transmittingNode.DoesContainConnection(recievingNode);
        // if(!doesConnectionExist)
        // {
        Arc connection = new Arc(this, transmittingNode, recievingNode, transferFunction);
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
        inputOperation.run();
        recieveSignals();
        sendInferenceSignals();
        outputOperation.run();
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

    public static <T> boolean hasIntersection(HashSet<T> s1, HashSet<T> s2) {
        HashSet<T> intersection = new HashSet<T>(s1);
        intersection.retainAll(s2);
        return intersection.size() > 0;
    }

}
