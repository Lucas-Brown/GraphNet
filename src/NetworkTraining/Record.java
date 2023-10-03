package src.NetworkTraining;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.stream.Stream;

import src.GraphNetwork.Local.Node;

/**
 * Remembers all signals that were recieved and sent from a particular node 
 */
public class Record implements Comparable<Record>{
    
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
     * The merged signal strength of the {@link currentNode} at the recorded time step
     */
    public final double nodeSignalStrength;

    /**
     * The output signal strength of the {@link currentNode} at the recorded time step
     */
    public final double nodeOutputStrength;

    /**
     * The index key of the current node for the incoming nodes
     */
    public final int incomingKey;

    public Record(Node currentNode, Collection<Node> incomingNodes, Collection<Node> outgoingNodes, double nodeSignalStrength, double nodeOutputStrength)
    {
        this.currentNode = currentNode;
        this.incomingNodes = new ArrayList<>(incomingNodes);
        this.outgoingNodes = new ArrayList<>(outgoingNodes);
        this.nodeSignalStrength = nodeSignalStrength;
        this.nodeOutputStrength = nodeOutputStrength;
        incomingKey = currentNode.nodeSetToBinStr(this.incomingNodes);
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

    public boolean hasNoOutputSignal()
    {
        return outgoingNodes.isEmpty();
    }

    @Override
    public String toString()
    {
        return "Record of " + currentNode.toString();
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

    @Override
    public int compareTo(Record o) {
        return currentNode.compareTo(o.currentNode);
    }

}
