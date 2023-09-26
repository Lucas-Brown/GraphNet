package src.GraphNetwork.Global;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.TreeSet;

import src.GraphNetwork.Local.ActivationFunction;
import src.GraphNetwork.Local.ActivationProbabilityDistribution;
import src.GraphNetwork.Local.Arc;
import src.GraphNetwork.Local.Node;
import src.NetworkTraining.History;

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

    /**
     * A list of all history objects that are currently in use
     */
    private final ArrayList<History> allActiveHistories;

    /**
     * maps a set of histories to all the nodes that are linked through their histories
     */
    private final HashMap<HashSet<History>, HashSet<Node>> historyMergers;

    public GraphNetwork()
    {
        networkData = new SharedNetworkData(1000, 0.2f, 0.9f, 1f);

        nodes = new ArrayList<>();
        activeNodes = new HashSet<>();
        activeNextNodes = new HashSet<>();
        errorNodes = new HashSet<>();
        allActiveHistories = new ArrayList<>();
        historyMergers = new HashMap<>();
    }

    /**
     * Creates a new node and adds it to the list of nodes within the network
     * TODO: Potentially remove external addition of nodes and connections in favor of dynamically adding nodes/edges during training
     * @return The node that was created 
     */
    public Node createNewNode(final ActivationFunction activationFunction)
    {
        Node n = new Node(this, networkData, activationFunction);
        nodes.add(n);
        return n;
    }

    /**
     * Creates a new signal and automattically alerts the network that the recieving node has been activated
     * @param sendingNode the node sending the signal
     * @param recievingNode the node recieving the signal
     * @param strength the value associated with the signal
     * @return a signal object containing the sending node, recieving node, and signal strength
     */
    public Signal createSignal(final Node sendingNode, final Node recievingNode, final double strength, History history)
    {
        activeNextNodes.add(recievingNode); // every time a signal is created, the network is notified of the reciever
        return new Signal(sendingNode, recievingNode, strength, history);
    }

    public Signal createSignal(final Node sendingNode, final Node recievingNode, final double strength)
    {
        return createSignal(sendingNode, recievingNode, strength, null);
    }

    /**
     * Creates a new history object and tells the network to track it
     * @return an empty history object 
     */
    public History createHistory()
    {
        History newHist = new History();
        allActiveHistories.add(newHist);
        return newHist;
    }


    /**
     * Step the entire network forward one itteration
     * Does not train the network
     */
    public void step()
    {
        recieveSignals();
        transmitSignals();
    }

    /**
     * 
     */
    public void trainingStep()
    {
        recieveSignals();
        


        mergeAllHistories();
        transmitSignals();
        advanceHistory();
    }

    /**
     * Tell all active nodes to accept all incoming signals
     */
    public void recieveSignals()
    {
        activeNodes = activeNextNodes;
        activeNodes.forEach(Node::acceptIncomingSignals);
        activeNextNodes = new HashSet<>();
    }

    /**
     * Tell each node to compute next phase of signals
     */
    public void transmitSignals()
    {
        // Will automatically collect generated signals in {@code activeNextNodes}
        activeNodes.stream().forEach(Node::attemptSendOutgoingSignals);
    }

    /**
     * Nodes should call this method to alert the network that two histories need to be merged.
     * @param historiesToMerge
     * @param alertingNode
     */
    public void alertHistoryMerge(HashSet<History> historiesToMerge, Node alertingNode)
    {
        // if the key already exists, the alerting node just needs to be added
        HashSet<Node> nodes = historyMergers.get(historiesToMerge); 
        if(nodes != null)
        {
            nodes.add(alertingNode);
            return;
        }

        // since the key was not found, collect all keys which contain any of the histories within historiesToMerge
        List<HashSet<History>> badKeys = historyMergers.keySet().stream()
            .filter(histSet -> hasIntersection(histSet, historiesToMerge))
            .toList();

        // if there are no keys which contain any element in historiesToMerge, then we only need to create a new element
        // otherwise, all keys and values need to be removed and unioned
        if(badKeys.size() == 0)
        {
            HashSet<Node> nodeList = new HashSet<>();
            nodeList.add(alertingNode);
            historyMergers.put(historiesToMerge, nodeList);
        }
        else
        {
            HashSet<Node> nodesInMerge = new HashSet<>();
            nodesInMerge.add(alertingNode);
            badKeys.forEach(key -> 
            {
                historiesToMerge.addAll(key);
                nodesInMerge.addAll(historyMergers.remove(key));
            });
            
            historyMergers.put(historiesToMerge, nodesInMerge);
        }
    }

    /**
     * Merge every history that has been queued
     */
    public void mergeAllHistories()
    {
        /**
         * I'm anticipating that each merger might take a while as they can get pretty big...
         */
        historyMergers.keySet().parallelStream()
            .forEach(historiesToMerge -> 
            {
                History mergedHistory = History.mergeHistories(historiesToMerge); // Merge the histories
                allActiveHistories.removeAll(historiesToMerge); // remove the merged histories 
                allActiveHistories.add(mergedHistory); // add the new merged history
                historyMergers.get(historiesToMerge) // set every node to have their proper history 
                    .stream()
                    .forEach(node -> node.setHistory(mergedHistory));
            });
    }

    /**
     * collect all records and step all histories 
     */
    public void advanceHistory()
    {
        gatherRecords();
        allActiveHistories.stream().forEach(History::step);
    }

    public void gatherRecords()
    {
        activeNodes.forEach(Node::addRecordToHistory);
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

    public static <T> boolean hasIntersection(HashSet<T> s1, HashSet<T> s2)
    {
        HashSet<T> intersection = new HashSet<T>(s1);
        intersection.retainAll(s2);
        return intersection.size() > 0;
    }
    

}
