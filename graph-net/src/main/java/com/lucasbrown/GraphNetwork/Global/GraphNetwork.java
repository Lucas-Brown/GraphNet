package com.lucasbrown.GraphNetwork.Global;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
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
 * Current idea for training: 
 * 1) record every node (and respective signal strength) that a chain of signals take
 * 2) if that chain reaches an output node, there's 2 possible scenarios:
 * 2a) the output node is supposed to have a value at that time, thus correcting the node is simply the process of backpropagation and reinforce firing probabilities
 * 2b) the output node was not supposed to have a value at that time, thus all firing probabilities need to be diminished
 * 3) if the chain doesn't reach the end, then the output node needs to send a signal back through the network to make a connection and create an error signal
 * 
 * In some scenarios, it might not matter the exact timing of when the signal reaches an output.
 * In such cases, training may involve uniformly increasing the likelyhood of firing until any signal reaches an output.
 */
public class GraphNetwork {

    private boolean isTraining;

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
     * A hash set containing every node that recieved an error propogation signal this step
     */
    private final HashSet<Node> errorNodes;

    /**
     * An operation which is to be defined by the user to set the values of input nodes
     */
    private Runnable inputOperation;
    
    /**
     * An operation which is to be defined by the user to correct the values of output nodes during training or get output data
     */
    private Runnable outputOperation;

    public GraphNetwork()
    {
        // TODO: remove hardcoding
        networkData = new SharedNetworkData(new ErrorFunction.MeanSquaredError(), 0.1);

        nodes = new ArrayList<>();
        activeNodes = new HashSet<>();
        activeNextNodes = new HashSet<>();
        errorNodes = new HashSet<>();
        inputOperation = () -> {};
        outputOperation = () -> {};
    }

    public void setInputOperation(Runnable inputOperation)
    {
        this.inputOperation = inputOperation == null ? () -> {} : inputOperation;
    }

    public void setOutputOperation(Runnable outputOperation)
    {
        this.outputOperation = outputOperation == null ? () -> {} : outputOperation;
    }

    /**
     * Creates a new node and adds it to the list of nodes within the network
     * TODO: Potentially remove external addition of nodes and connections in favor of dynamically adding nodes/edges during training
     * @return The node that was created 
     */
    public Node createHiddenNode(final ActivationFunction activationFunction)
    {
        Node n = new Node(this, networkData, activationFunction);
        nodes.add(n);
        return n;
    }

    public InputNode createInputNode(final ActivationFunction activationFunction)
    {
        InputNode n = new InputNode(this, networkData, activationFunction);
        nodes.add(n);
        return n;
    }

    public OutputNode createOutputNode(final ActivationFunction activationFunction)
    {
        OutputNode n = new OutputNode(this, networkData, activationFunction);
        nodes.add(n);
        return n;
    }

    public void addNewConnection(Node transmittingNode, Node recievingNode, ActivationProbabilityDistribution transferFunction)
    {
        //boolean doesConnectionExist = transmittingNode.DoesContainConnection(recievingNode);
        //if(!doesConnectionExist)
        //{
        Arc connection = new Arc(this, transmittingNode, recievingNode, transferFunction);
        transmittingNode.addOutgoingConnection(connection);
        recievingNode.addIncomingConnection(connection);
        //}
        //return doesConnectionExist;
    }

    /**
     * Creates a new signal and automattically alerts the network that the recieving node has been activated
     * @param sendingNode the node sending the signal
     * @param recievingNode the node recieving the signal
     * @param strength the value associated with the signal
     * @return a signal object containing the sending node, recieving node, and signal strength
     */
    public Signal createSignal(final Node sendingNode, final Node recievingNode, final double strength)
    {
        activeNextNodes.add(recievingNode); // every time a signal is created, the network is notified of the reciever
        return new Signal(sendingNode, recievingNode, strength);
    }


    /**
     * Step the entire network forward one itteration
     * Does not train the network
     */
    public void step()
    {
        inputOperation.run();
        recieveSignals();
        outputOperation.run();
        transmitSignals();
        deactivateNodes();
    }

    /**
     * 
     */
    public void trainingStep()
    {
        inputOperation.run();
        outputOperation.run();
        recieveSignals();
        
        transmitSignals();

        deactivateNodes();
    }

    public boolean isNetworkDead()
    {
        return activeNodes.isEmpty();
    }

    /**
     * Tell all active nodes to accept all incoming signals
     */
    public void recieveSignals()
    {
        activeNodes = activeNextNodes;
        activeNextNodes = new HashSet<>();
        activeNodes.forEach(Node::activate);
        activeNodes.forEach(Node::acceptSignals);
    }

    /**
     * Tell each node to compute next phase of signals
     */
    public void transmitSignals()
    {
        // Will automatically collect generated signals in {@code activeNextNodes}
        activeNodes.stream().forEach(Node::attemptSendOutgoingSignals);
    }

    public void deactivateNodes()
    {
        activeNodes.stream().forEach(Node::deactivate);
    }

    public String allActiveNodesString()
    {
        TreeSet<Node> sSet = new TreeSet<Node>(activeNodes);
        StringBuilder sb = new StringBuilder();
        sSet.forEach(node -> 
        {
            sb.append(node.toString());
            sb.append('\t');
        });
        return sb.toString();
    }

    public static <T> boolean hasIntersection(HashSet<T> s1, HashSet<T> s2)
    {
        HashSet<T> intersection = new HashSet<T>(s1);
        intersection.retainAll(s2);
        return intersection.size() > 0;
    }
    

}
