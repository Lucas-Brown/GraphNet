package src.NetworkTraining;

import java.util.Arrays;

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
    private final Node[] incomingNodes;

    /**
     * All nodes that the {@link currentNode} sent signals to
     */
    private final Node[] outgoingNodes;
    
    /**
     * The output signal strength of the {@link currentNode} at the recorded time step
     */
    public final float nodeSignalStrength;

    public Record(Node currentNode, Node[] incomingNodes, Node[] outgoingNodes, float nodeSignalStrength)
    {
        this.currentNode = currentNode;
        this.incomingNodes = incomingNodes;
        this.outgoingNodes = outgoingNodes;
        this.nodeSignalStrength = nodeSignalStrength;
    }
    
    public Node getCurrentNode() {
        return currentNode;
    }

    public Node[] getIncomingNodes() {
        return Arrays.copyOf(incomingNodes, incomingNodes.length);
    }

    public Node[] getOutgoingNodes() {
        return Arrays.copyOf(outgoingNodes, outgoingNodes.length);
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

}
