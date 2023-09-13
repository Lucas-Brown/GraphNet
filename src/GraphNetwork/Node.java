package src.GraphNetwork;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

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
     * All incoming and outgoing node connections. 
     */
    private ArrayList<NodeConnection> incoming, outgoing;

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


    public Node()
    {
        incoming = new ArrayList<NodeConnection>();
        outgoing = new ArrayList<NodeConnection>();
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
    boolean AddIncomingConnection(NodeConnection connection)
    {
        return incoming.add(connection);
    }

    /**
     * Add an outgoing connection to the node
     * @param connection
     * @return true
     */
    boolean AddOutgoingConnection(NodeConnection connection)
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
    public void CorrectRecievingValue(float epsilon)
    {
        final int signal_count = incomingSignals.size();
        for(Signal signal: incomingSignals)
        {
            signal.recievingFunction.AdjustSignalStrength(mergedSignal, epsilon/signal_count);
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
        outgoing.forEach(connection -> {
            if(connection.SendSignal(mergedSignal) != null)
            {
                signaledNodes.add(connection.recieving);
            };
        });
        return signaledNodes.stream();
    }

    /**
     * Attempt to send an error signal back to every incoming node.
     * Updates the transfer function based on the error of the signal.
     * @param epsilon 
     * @return a stream containing every node that was sent an error signal
     */
    public Stream<Node> TransmitError(Integer N_Limiter, float epsilon)
    {
        HashMap<NodeConnection, Float> signalMap = new HashMap<>();
        HashSet<NodeConnection> activatedErrorConnections = new HashSet<>();

        errorSignals.forEach(errorSignal -> 
        {
            // If the recieving function is null, then the node was manually set and mergedSignal contains the target
            // Otherwise, the strength of the signal is the target and the recieving function estimates the strength of a signal
            
            HashSet<NodeConnection> errorConnections;
            
            if(errorSignal.recievingFunction == null)
            {
                errorConnections = GetErrorConnections(activatedErrorConnections, mergedSignal);
            }
            else
            {
                errorConnections = GetErrorConnections(activatedErrorConnections, errorSignal.strength);
            }

            if(errorConnections.isEmpty()) return;
            CalculateRecievedError(signalMap, errorConnections);
        });

        errorSignals.clear();

        // Send the error to each respective node
        signalMap.forEach((connection, strength) -> 
        {
            connection.recieving.NotifyErrorSignal(new Signal(connection.transferFunc, strength));
        });


        // Correct each transfer function 
        signalMap.forEach((connection, strength) -> 
        {
            connection.transferFunc.AdjustSignalStrength(strength, epsilon);
        } );

        return activatedErrorConnections.stream().map(connection -> connection.sending);
    }

    private HashSet<NodeConnection> GetErrorConnections(HashSet<NodeConnection> activatedErrorConnections, float target)
    {
        HashSet<NodeConnection> errorConnections = incoming.stream()
            .filter(connection -> connection.transferFunc.ShouldSend(target)) 
            .collect(Collectors.toCollection(HashSet::new));

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
    private void CalculateRecievedError(HashMap<NodeConnection, Float> signalMap, HashSet<NodeConnection> errorConnections)
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
    public void ReinforceSignalPathways(int N_Limiter)
    {
        for(Signal signal: outgoingSignals)
        {
            signal.recievingFunction.UpdateDistribution(mergedSignal, N_Limiter);
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
