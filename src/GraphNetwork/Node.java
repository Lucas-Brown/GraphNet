package src.GraphNetwork;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.stream.Stream;

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of sending a signal out to other connected nodes
 */
public class Node {

    /**
     * All incoming and outgoing node connections. 
     */
    private ArrayList<NodeConnection> incoming, outgoing;

    /**
     * All incoming and outgoing signals 
     */
    private ArrayList<Signal> incomingSignals, outgoingSignals;

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
     * 
     * @param target
     */
    public void SetNodeSignal(HashSet<Node> signaledNodes, float target)
    {
        mergedSignal = target;
        signaledNodes.add(this);
    }

    /**
     * The network has been given a data point which is known to be correct and needs to update accordingly.
     * This is EXPLICITLY for when a signal is expected and it's corresponding value is incorrect.
     * If no node is currently sending a signal to this node, then all reciving nodes are updated.
     * If at least one node has sent a signal to this node, calculate the corresponding distribution of error
     * @param target
     */
    public void RecieveCorrectionSignal(float target)
    {
        switch (incomingSignals.size()) {
            case 0:
                
                break;
        
            default:
                break;
        }
    }

    /**
     * Notify this node of a new incoming signal
     * @param signal The value of the incoming signal
     */
    public void RecieveSignal(Signal signal)
    {
        incomingSignals.add(signal);
    }

    /**
     * Handle all incoming signals and store the resulting strength
     * TODO: consider more approaches to merging multiple incoming signals such as log(exp(x1) + exp(x2)) and generalize
     */
    public void HandleIncomingSignals()
    {
        mergedSignal = 0;
        for(Signal signal : incomingSignals)
        {
            mergedSignal += signal.strength;
        }
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
            Signal signal = connection.SendSignal(mergedSignal);
            if(signal != null)
            {
                outgoingSignals.add(signal);
                signaledNodes.add(connection.recieving);
            };
        });
        return signaledNodes.stream();
    }

    
    public void ReinforceSignalPathways(int N_estimator)
    {
        for(Signal signal: outgoingSignals)
        {
            signal.recievingFunction.UpdateDistribution(signal.strength, N_estimator);
        }
        outgoingSignals.clear();
    }


    public String ToVisualString()
    {
        if(incomingSignals.isEmpty())
        {
            return "O"; 
        }
        else
        {
            return Float.toString(mergedSignal);
        }
    }

}
