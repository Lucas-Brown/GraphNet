package src.NetworkTraining;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.stream.Stream;

import src.GraphNetwork.Local.Node;

/**
 * Remembers all signals that were recieved and sent from a particular node 
 */
public class Record {
    
    /**
     * The current node in this history
     */
    public final Node currentNode;

    /**
     * All nodes that the {@link currentNode} recieved signals from
     */
    final ArrayList<Node> incomingNodes;

    /**
     * All nodes that the {@link currentNode} sent signals to
     */
    final ArrayList<Node> outgoingNodes;
    
    /**
     * The output signal strength of the {@link currentNode} at the recorded time step
     */
    public final double nodeSignalStrength;

    public Record(Node currentNode, Collection<Node> incomingNodes, Collection<Node> outgoingNodes, double nodeSignalStrength)
    {
        this.currentNode = currentNode;
        this.incomingNodes = new ArrayList<>(incomingNodes);
        this.outgoingNodes = new ArrayList<>(outgoingNodes);
        this.nodeSignalStrength = nodeSignalStrength;
    }
    
    public Node getCurrentNode() {
        return currentNode;
    }

    public ArrayList<Node> getIncomingNodes() {
        return new ArrayList<>(incomingNodes);
    }

    public ArrayList<Node> getOutgoingNodes() {
        return new ArrayList<>(outgoingNodes);
    }

    Stream<Node> getIncomingNodesStream() {
        return incomingNodes.stream();
    }

    Stream<Node> getOutgoingNodesStream() {
        return outgoingNodes.stream();
    }

    public boolean hasOutputSignal()
    {
        return !outgoingNodes.isEmpty();
    }

    @Override
    public String toString()
    {
        return currentNode.toString();
    }

    @Override
    public int hashCode()
    {
        return currentNode.hashCode();
    }

    /**
     * This equals will also accept a {@code Node} object  
     */
    @Override
    public boolean equals(Object o)
    {
        if(o instanceof Record)
        {
            return hashCode() == ((Record) o).hashCode();
        }
        else if(o instanceof Node)
        {
            return hashCode() == ((Node) o).hashCode();
        }
        return false;
    }

}
