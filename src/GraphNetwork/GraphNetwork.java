package src.GraphNetwork;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.TreeSet;
import java.util.stream.Collectors;

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
    public Node CreateNewNode()
    {
        Node n = new Node(networkData);
        nodes.add(n);
        return n;
    }

    /**
     * Tell all active nodes to accept all incoming signals
     */
    public void RecieveSignals()
    {
        activeNodes = activeNextNodes;
        activeNodes.forEach(Node::HandleIncomingSignals);
        activeNextNodes = new HashSet<>();
        errorNodes = new HashSet<>();
    }

    /**
     * Correct the value of a node. Sets the value of that node to target and sends an error signal through the network.
     * To directly set the value of a ndoe without sending an error signal see {@link DirectSetNodeValue}.
     * @param node The node to correct
     * @param target The value of the node
     */
    public void CorrectNodeValue(Node node, float target)
    {
        node.CorrectNodeValue(target);
        activeNodes.add(node);
        errorNodes.add(node);
    }


    /**
     * Directly set the value of a node. Should be used to input data into the network.
     * For correcting the output of a node see {@link CorrectNodeValue}.
     * @param node The node to assign the value to.
     * @param value The value to se the node to.
     * @return Whether the node was already active.
     */
    public boolean DirectSetNodeValue(Node node, float value)
    {
        node.DirectSetNodeValue(value);
        return activeNodes.add(node);
    }

    /**
     * Tell each node to compute next phase of signals
     */
    public void TransmitSignals()
    {
        activeNextNodes = activeNodes.stream().flatMap(Node::TransmitSignal).collect(Collectors.toCollection(HashSet::new));
    }

    /**
     * Reinforce the strength of each signal and more closely allign the recieved value to the mean probability. 
     */
    public void ReinforceSignals()
    {
        activeNodes.forEach(Node::CorrectRecievingValue);
        activeNodes.forEach(Node::ReinforceSignalPathways);
    }

    /**
     * Propogate all error signals and adjust transfer functions accordingly.
     */
    public void PropagateErrors()
    {
        errorNodes = errorNodes.stream().flatMap(Node::TransmitError).collect(Collectors.toCollection(HashSet::new));
    }


    public void AddNewConnection(Node transmittingNode, Node recievingNode, NodeTransferFunction transferFunction)
    {
        //boolean doesConnectionExist = transmittingNode.DoesContainConnection(recievingNode);
        //if(!doesConnectionExist)
        //{
        Edge connection = new Edge(transmittingNode, recievingNode, transferFunction);
        transmittingNode.AddOutgoingConnection(connection);
        recievingNode.AddIncomingConnection(connection);
        //}
        //return doesConnectionExist;
    }

    public String AllActiveNodesString()
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

    /**
     * A collection of data which modifies the training and firing rates of each node.
     * All nodes are given access to this data to use but modification should be strictly controlled by {@link GraphNetwork}
     */
    public class SharedNetworkData
    {

        /**
         * Datapoint limiter for the number of data points each distribution represents
         */
        private int N_Limiter; 

        /**
         * Step size for adjusting the output values of nodes
         */
        private float epsilon;

        /**
         * The factor of decay for the likelyhood of a node firing sucessive signals in one step
         * i.e. The first check is unchanged, the second check is multiplied by a factor of likelyhoodDecay, the third a factor of likelyhoodDecay * likelyhoodDecay and so on.
         */
        private float likelyhoodDecay;

        /**
         * Dynamically adjusts the firing rate of the network  
         */
        private float globalFiringRateMultiplier;

        private SharedNetworkData(int N_Limiter, float epsilon, float likelyhoodDecay, float globalFiringRateMultiplier)
        {
            this.N_Limiter = N_Limiter;
            this.epsilon = epsilon;
            this.likelyhoodDecay = likelyhoodDecay;
            this.globalFiringRateMultiplier = globalFiringRateMultiplier;
        }

        public int getN_Limiter() {
            return N_Limiter;
        }

        public float getEpsilon() {
            return epsilon;
        }

        public float getLikelyhoodDecay() {
            return likelyhoodDecay;
        }

        public float getGlobalFiringRateMultiplier() {
            return globalFiringRateMultiplier;
        }
    }

}
