package src.GraphNetwork.Global;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.TreeSet;
import java.util.stream.Collectors;

import src.GraphNetwork.Local.ActivationProbabilityDistribution;
import src.GraphNetwork.Local.Arc;
import src.GraphNetwork.Local.Node;

/**
 * A neural network using a probabalistic directed graph representation.
 * Training currently a work in progress
 */
public class GraphNetwork {

    /**
     * An object to encapsulate all network hyperparameters
     */
    private final SharedNetworkData networkData;

    /**
     * A list of all nodes within the graph network
     */
    private ArrayList<Node> nodes;

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
    private HashSet<Node> errorNodes;

    public GraphNetwork()
    {
        networkData = new SharedNetworkData(1000, 0.2f, 0.9f, 1f);

        nodes = new ArrayList<>();
        activeNodes = new HashSet<>();
        activeNextNodes = new HashSet<>();
        errorNodes = new HashSet<>();
    }

    /**
     * Creates a new node and adds it to the list of nodes within the network
     * TODO: Potentially remove external addition of nodes and connections in favor of dynamically adding nodes/edges during training
     * @return The node that was created 
     */
    public Node createNewNode()
    {
        Node n = new Node(this, networkData);
        nodes.add(n);
        return n;
    }

    public Signal createSignal(final Node sendingNode, final Node recievingNode, final float strength)
    {
        activeNextNodes.add(recievingNode); // every time a signal is created, the network is notified of the reciever
        return new Signal(sendingNode, recievingNode, strength);
    }

    /**
     * Tell all active nodes to accept all incoming signals
     */
    public void recieveSignals()
    {
        activeNodes = activeNextNodes;
        activeNodes.forEach(Node::handleIncomingSignals);
        activeNextNodes = new HashSet<>();
        errorNodes = new HashSet<>();
    }

    /**
     * Tell each node to compute next phase of signals
     */
    public void transmitSignals()
    {
        //activeNextNodes = activeNodes.stream().flatMap(Node::TransmitSignal).collect(Collectors.toCollection(HashSet::new));
    }

    /**
     * Reinforce the strength of each signal and more closely allign the recieved value to the mean probability. 
     */
    public void reinforceSignals()
    {
        //activeNodes.forEach(Node::CorrectRecievingValue);
        //activeNodes.forEach(Node::ReinforceSignalPathways);
    }

    /**
     * Propogate all error signals and adjust transfer functions accordingly.
     */
    public void propagateErrors()
    {
        //errorNodes = errorNodes.stream().flatMap(Node::TransmitError).collect(Collectors.toCollection(HashSet::new));
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


    

}
