package src.GraphNetwork.Local;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Objects;
import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

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
    private final SharedNetworkData networkData;

    /**
     * All incoming and outgoing node connections. 
     */
    private final ArrayList<Arc> incoming, outgoing;

    /**
     * All incoming signals 
     */
    private final ArrayList<Signal> incomingSignals;

    
    /**
     * All outgoing signals 
     */
    private final ArrayList<Signal> outgoingSignals;

    
    /**
     * All error signals 
     */
    private final ArrayList<Signal> errorSignals;

    /**
     * Maps all incoming node ID's to an int from 0 to the number of incoming nodes -1
     */
    private HashMap<Integer, Integer> orderedIDMap;

    /**
     * Each possible combinations of inputs has a corresponding unique set of weights and biases
     * both grow exponentially, which is bad, but every node should have relatively few connections 
     */
    private float[][] weights;
    private float[] biases;

    /**
     * The signal strength for the current iteration
     */
    private float mergedSignal;

    public Node(final GraphNetwork network, final SharedNetworkData networkData)
    {
        this.network = Objects.requireNonNull(network);
        this.networkData = Objects.requireNonNull(networkData);
        incoming = new ArrayList<Arc>();
        outgoing = new ArrayList<Arc>();
        incomingSignals = new ArrayList<>();
        outgoingSignals = new ArrayList<>();
        errorSignals = new ArrayList<>();
        weights = new float[1][1];
        biases = new float[0];
        id = ID_COUNTER++;
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
            biases[i] = rand.nextFloat(); 

            // populate the weights array
            int count = weights[i].length + 1;
            weights[i] = new float[count];
            for(int j = 0; j < count; j++)
            {
                weights[i][j] = rand.nextFloat();
            }
        }
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
    void recieveSignal(Signal signal)
    {
        incomingSignals.add(signal);
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
     * Handle all incoming signals and store the resulting strength
     */
    public void handleIncomingSignals()
    {
        if(incomingSignals.isEmpty()) return;
       // mergedSignal = MergeSignal(incomingSignals.stream().mapToDouble(Signal::GetOutputStrength));
        
    }


    @Override
    public String toString()
    {
        return "node " + Integer.toString(id) + ": " + Float.toString(mergedSignal);
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
