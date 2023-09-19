package src.GraphNetwork;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;
import src.GraphNetwork.GraphNetwork.SharedNetworkData;

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
     * The network hyperparameters  
     */
    private final SharedNetworkData networkData;

    /**
     * All incoming and outgoing node connections. 
     */
    private ArrayList<Edge> incoming, outgoing;

    /**
     * All incoming signals 
     */
    private ArrayList<Signal> incomingSignals;

    
    /**
     * All outgoing signals 
     */
    private ArrayList<Signal> outgoingSignals;

    
    /**
     * All error signals 
     */
    private ArrayList<Signal> errorSignals;

    /**
     * The signal strength for the current iteration
     */
    private float mergedSignal;


    Node(final SharedNetworkData networkData)
    {
        this.networkData = networkData;
        incoming = new ArrayList<Edge>();
        outgoing = new ArrayList<Edge>();
        incomingSignals = new ArrayList<>();
        outgoingSignals = new ArrayList<>();
        errorSignals = new ArrayList<>();
        id = ID_COUNTER++;
    }

    /**
     * 
     * @param node
     * @return whether this node is connected to the provided node
     */
    public boolean DoesContainConnection(Node node)
    {
        return outgoing.stream().anyMatch(connection -> connection.DoesMatchNodes(this, node));
    }

    
    /**
     * Correct the value of this node and send an error signal through the network.
     * 
     * @param target the value this node should have at this step
     */
    void CorrectNodeValue(float target)
    {
        // a null recieving function will indicate that the mergedSignal contains the target
        errorSignals.add(new Signal(null, mergedSignal)); 
        mergedSignal = target;
    }

    /**
     * Directly set the value of this node
     * 
     * @param value the value this node should have at this step
     */
    void DirectSetNodeValue(float value)
    {
        mergedSignal = value;
    }

    /**
     * Add an incoming connection to the node
     * @param connection
     * @return true
     */
    boolean AddIncomingConnection(Edge connection)
    {
        return incoming.add(connection);
    }

    /**
     * Add an outgoing connection to the node
     * @param connection
     * @return true
     */
    boolean AddOutgoingConnection(Edge connection)
    {
        return outgoing.add(connection);
    }


    /**
     * Notify this node that it has recieved an error signal
     * @param error
     */
    void NotifyErrorSignal(Signal signal)
    {
        errorSignals.add(signal);
    }

    /**
     * Notify this node of a new incoming signal
     * @param signal The value of the incoming signal
     */
    void NotifyRecieveSignal(Signal signal)
    {
        incomingSignals.add(signal);
    }

    
    /**
     * Notify this node of a new incoming signal
     * @param signal The value of the incoming signal
     */
    void NotifyTransmittingSignal(Signal signal)
    {
        outgoingSignals.add(signal);
    }


    /**
     * Handle all incoming signals and store the resulting strength
     */
    public void HandleIncomingSignals()
    {
        if(incomingSignals.isEmpty()) return;
        mergedSignal = MergeSignal(incomingSignals.stream().mapToDouble(Signal::GetOutputStrength));
        
    }

    /**
     * TODO: consider more approaches to merging multiple incoming signals such as log(exp(x1) + exp(x2)) and generalize
     * @param strengths
     * @return
     */
    private float MergeSignal(DoubleStream strengths)
    {
        return (float) strengths.average().getAsDouble();
    }


    /**
     * Adjust the signal sent to this node
     */
    public void CorrectRecievingValue()
    {
        final int signal_count = incomingSignals.size();
        for(Signal signal: incomingSignals)
        {
            signal.recievingFunction.AdjustSignalStrength(mergedSignal, networkData.getEpsilon()/signal_count);
        }
        incomingSignals.clear();
    }


    /**
     * Attempt to send a signal out to every outward connecting neuron
     * @return a stream containing every node that was sent a signal
     */
    public Stream<Node> TransmitSignal()
    {
        HashSet<Node> signaledNodes = new HashSet<>(outgoing.size());
        float factor = 1f;
        final float decay = networkData.getLikelyhoodDecay();

        Collections.shuffle(outgoing); // shuffle to ensure no connection has an order-dependent advantage
        for(Edge connection : outgoing)
        {
            if(connection.SendSignal(mergedSignal, factor) != null)
            {
                signaledNodes.add(connection.recieving);
            };
            factor *= decay;
        };
        return signaledNodes.stream();
    }

    /**
     * Attempt to send an error signal back to every incoming node.
     * Updates the transfer function based on the error of the signal.
     * @param epsilon 
     * @return a stream containing every node that was sent an error signal
     */
    public Stream<Node> TransmitError()
    {
        HashMap<Edge, Float> signalMap = new HashMap<>();
        HashSet<Edge> activatedErrorConnections = new HashSet<>();

        errorSignals.forEach(errorSignal -> 
        {
            // If the recieving function is null, then the node was manually set and mergedSignal contains the target
            // Otherwise, the strength of the signal is the target and the recieving function estimates the strength of a signal
            
            HashSet<Edge> errorConnections;
            
            if(errorSignal.recievingFunction == null)
            {
                errorConnections = GetErrorConnections(activatedErrorConnections, mergedSignal);
            }
            else
            {
                errorConnections = GetErrorConnections(activatedErrorConnections, errorSignal.strength);
                errorSignal.recievingFunction.UpdateDistribution(errorSignal.strength, networkData.getN_Limiter());
            }

            if(errorConnections.isEmpty()) return;
            CalculateRecievedError(signalMap, errorConnections);
        });


        // Send the error to each respective node
        signalMap.forEach((connection, strength) -> 
        {
            connection.recieving.NotifyErrorSignal(new Signal(connection.transferFunc, connection.transferFunc.GetMostLikelyValue()));
        });


        // Correct each transfer function 
        signalMap.forEach((connection, strength) -> 
        {
            connection.transferFunc.AdjustSignalStrength(strength, networkData.getEpsilon());
        } );

        errorSignals.clear();
        return activatedErrorConnections.stream().map(connection -> connection.sending);
    }

    private HashSet<Edge> GetErrorConnections(HashSet<Edge> activatedErrorConnections, float target)
    {
        float factor = 1f; 
        final float decay = networkData.getLikelyhoodDecay();

        HashSet<Edge> errorConnections = new HashSet<>(incoming.size());
        for(Edge connection : incoming)
        {
            if(connection.transferFunc.ShouldSend(target, factor))
            {
                errorConnections.add(connection);
            }
            factor *= decay;
        }

        errorConnections.forEach(connection -> System.out.println("Transfer mean: " + connection.transferFunc.GetMostLikelyValue() + "\t Target: " + target));
        
        activatedErrorConnections.addAll(errorConnections);
        return errorConnections;
    }

    /**
     * 
     * @param signalMap
     * @param activatedErrorConnections
     * @param errorSignal
     * @param epsilon
     */
    private void CalculateRecievedError(HashMap<Edge, Float> signalMap, HashSet<Edge> errorConnections)
    {
        // compute total contribution from each sucessful signal
        float cumulativeSignal = MergeSignal(errorConnections.stream().mapToDouble(connection -> connection.transferFunc.strength));

        errorConnections.stream().forEach(connection ->
        {
            Float cumulativeError = signalMap.getOrDefault(connection, new Float(0));
            cumulativeError += cumulativeSignal - connection.transferFunc.strength;
        });
    }

    /**
     * Reinforce the distribution towards the sent signal
     * @param N_Limiter
     */
    public void ReinforceSignalPathways()
    {
        for(Signal signal: outgoingSignals)
        {
            signal.recievingFunction.UpdateDistribution(mergedSignal, networkData.getN_Limiter());
        }
        outgoingSignals.clear();
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

}
