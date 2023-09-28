package src.GraphNetwork.Local;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Objects;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import src.NetworkTraining.History;
import src.NetworkTraining.Record;
import src.GraphNetwork.Global.GraphNetwork;
import src.GraphNetwork.Global.Signal;
import src.GraphNetwork.Global.SharedNetworkData;

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of sending a signal out to other connected nodes
 */
public class Node implements Comparable<Node>{

    /**
     * The coutner is used to give each node a unique ID 
     */
    private static int ID_COUNTER = 0;

    /**
     * A unique identifying number for this node.
     */
    public final int id;

    /**
     * The network that this node belongs to
     */
    public final GraphNetwork network;

    /**
     * The network hyperparameters  
     */
    protected final SharedNetworkData networkData;

    /**
     * The activation function 
     */
    protected final ActivationFunction activationFunction;

    /**
     * All incoming and outgoing node connections. 
     */
    protected final ArrayList<Arc> incoming, outgoing;

    /**
     * All incoming signals 
     */
    protected final ArrayList<Signal> incomingSignals;

    /**
     * All outgoing signals 
     */
    protected final ArrayList<Signal> outgoingSignals;

    /**
     * All error signals 
     */
    protected final ArrayList<Signal> errorSignals;

    /**
     * The current history that has reached this node if available
     */
    protected History history;

    /**
     * Maps all incoming node ID's to an int from 0 to the number of incoming nodes -1
     */
    protected HashMap<Integer, Integer> orderedIDMap;

    /**
     * Each possible combinations of inputs has a corresponding unique set of weights, biases, and error signals
     * both grow exponentially, which is bad, but every node should have relatively few connections 
     */
    private double[][] weights;
    private double[] biases;
    private double[] errorSignal;

    /**
     * The signal strength for the current iteration
     */
    protected double mergedSignal;

    /**
     * Holds the accumulated error signal to this node for training.
     */
    protected double accumulatedError;

    public Node(final GraphNetwork network, final SharedNetworkData networkData, final ActivationFunction activationFunction)
    {
        id = ID_COUNTER++;
        this.network = Objects.requireNonNull(network);
        this.networkData = Objects.requireNonNull(networkData);
        this.activationFunction = activationFunction;
        incoming = new ArrayList<Arc>();
        outgoing = new ArrayList<Arc>();
        incomingSignals = new ArrayList<>();
        outgoingSignals = new ArrayList<>();
        errorSignals = new ArrayList<>();
        weights = new double[1][1];
        biases = new double[0];
        accumulatedError = 0;
    }

    /**
     * 
     * @param node
     * @return whether this node is connected to the provided node
     */
    public boolean doesContainConnection(Node node)
    {
        return outgoing.stream().anyMatch(connection -> connection.doesMatchNodes(this, node));
    }

    /**
     * Get the arc associated with the transfer from this node to the given recieving node
     * @param recievingNode
     * @return The arc if present, otherwise null
     */
    public Arc getArc(Node recievingNode)
    {
        return outgoing.stream()
            .filter(arc -> arc.doesMatchNodes(this, recievingNode))
            .findAny()
            .orElse(null);
    }

    /**
     * Add an incoming connection to the node
     * @param connection
     * @return true
     */
    public boolean addIncomingConnection(Arc connection)
    {
        orderedIDMap.put(connection.sending.id, 1 << orderedIDMap.size());
        appendWeightsAndBiases();
        return incoming.add(connection);
    }

    /**
     * Add an outgoing connection to the node
     * @param connection
     * @return true
     */
    public boolean addOutgoingConnection(Arc connection)
    {
        return outgoing.add(connection);
    }

    /**
     * Notify this node of a new incoming signal
     * @param signal The value of the incoming signal
     */
    void recieveSignal(Signal signal)
    {
        incomingSignals.add(signal);
    }

    /**
     * Notify this node that it has recieved an error signal
     * @param error
     */
    void recieveErrorSignal(Signal signal)
    {
        errorSignals.add(signal);
    }

    /**
     * Notify this node of a new incoming signal
     * @param signal The value of the incoming signal
     */
    void transmittingSignal(Signal signal)
    {
        outgoingSignals.add(signal);
    }

    /**
     * Adds another layer of depth to the weights and biases hyper array
     */
    private void appendWeightsAndBiases() 
    {
        Random rand = new Random();
        final int old_size = biases.length;
        final int new_size = 1 << orderedIDMap.size();

        // the first half doesn't need to be changed
        biases = Arrays.copyOf(biases, new_size);
        weights = Arrays.copyOf(weights, new_size);
        
        // the second half needs entirely new data
        for(int i = old_size; i < new_size; i++)
        {
            biases[i] = rand.nextDouble(); 

            // populate the weights array
            int count = weights[i].length + 1;
            weights[i] = new double[count];
            for(int j = 0; j < count; j++)
            {
                weights[i][j] = rand.nextDouble();
            }
        }
    }


    /**
     * Create a {@code Record} of the recieved and sent signals
     * @return a record
     */
    public Record generateStepRecord()
    {
        return new Record(this, 
        incomingSignals.stream().map(Signal::getSendingNode).toList(),
        outgoingSignals.stream().map(Signal::getRecievingNode).toList(),
        mergedSignal); 
    }


    /**
     * collect all histories from signals and notify the network if two histories are detected
     */
    public void collectHistoriesAndAlertMerge()
    {
        HashSet<History> histories = incomingSignals.stream()
            .map(signal -> signal.history)
            .filter(Objects::nonNull)
            .collect(Collectors.toCollection(HashSet::new));

        switch(histories.size())
        {
            case 0:
                history = null;
                break;
            case 1:
                history = histories.iterator().next();
                break;
            default:
                network.notifyHistoryMerge(histories, this);
        }

    }

    /**
     * History is whatever I say it is
     */
    public void setHistory(History history)
    {
        this.history = history;
    }

    /**
     * If this node has recieved a history object, a record is generated and added to the history
     */
    public void addRecordToHistory()
    {
        if(history != null)
        {
            history.addToCurrentRecord(generateStepRecord());
        }
    }

    /**
     * Handle all incoming signals and store the resulting strength
     */
    public void acceptIncomingSignals()
    {
        assert !incomingSignals.isEmpty() : "handleIncomingSignals should never be called if no signals have been recieved.";
        incomingSignals.sort((s1, s2) -> Integer.compare(s1.recievingNode.id, s2.recievingNode.id)); // sorting by id ensure that the weights are applied to the correct node/signal

        collectHistoriesAndAlertMerge();

        // map every recieving node id to its corresponding value and combine.
        // for example, an id of 6 may map to 0b0010 and an id of 2 may map to 0b1000
        // binary_string will thus contain the value 0b1010
        int binary_string = incomingSignals.stream()
            .mapToInt(signal -> orderedIDMap.get(signal.recievingNode.id)) 
            .reduce(0, (result, id_bit)  -> result &= id_bit); // effectively the same as a sum in this case

        // Use the binary_string to select which set of weights to apply 
        double[] input_weights = weights[binary_string];

        mergedSignal = IntStream.range(0, input_weights.length)
            .mapToDouble(i -> input_weights[i] * incomingSignals.get(i).strength)
            .sum();
            
        mergedSignal += biases[binary_string];

        mergedSignal = activationFunction.activator(mergedSignal);
        
    }

    public void attemptSendOutgoingSignals()
    {
        /**
         * Attempt to send a signal out to every outward connecting neuron
         * @return a stream containing every node that was sent a signal
         */
        double factor = 1f;
        final double decay = networkData.getLikelyhoodDecay();

        Collections.shuffle(outgoing); // shuffle to ensure no connection has an order-dependent advantage
        for(Arc connection : outgoing)
        {
            connection.sendSignal(mergedSignal, factor, history);
            factor *= decay;
        };
    }

    protected static void diminishFiringChances(History history, Node rootNode)
    {
        history.getNodeHistoryIterator(rootNode).forEachRemaining(recordList -> 
        {
            recordList.stream().forEach(Node::diminishDistributionOfRecord);
        });
    }

    private static void diminishDistributionOfRecord(Record record)
    {
        Node currentNode = record.currentNode;
        /*
         * Find the arc associated with the transfer between the current node and the output node
         * Then, diminish the probability of that node  
         */
        record.getOutgoingNodes().stream()
            .map(currentNode::getArc)
            .forEach(arc -> arc.probDist.diminishDistribution(record.nodeSignalStrength));
    }

    protected void correctSignalValue(double target)
    {
        if(history == null) return; // can't correct signal value without a full history back to an input node
        double mse = NetworkError.MSE(mergedSignal, target);

    }

    protected void sendErrorSignal()
    {

    }

    @Override
    public String toString()
    {
        return "node " + Integer.toString(id) + ": " + Double.toString(mergedSignal);
    }

    @Override
    public int hashCode()
    {
        return id;
    }

    @Override
    public int compareTo(Node o) {
        return id - o.id;
    }

    @Override
    public boolean equals(Object o) {
        if(!(o instanceof Node)) return false;
        return id == ((Node) o).id;
    }

}
