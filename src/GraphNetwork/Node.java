package src.GraphNetwork;

import java.util.ArrayList;

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of sending a signal out to other connected nodes
 */
public class Node {

    /**
     * All outgoing node connections. 
     */
    private ArrayList<NodeConnection> connections;

    public Node()
    {
        connections = new ArrayList<NodeConnection>();
    }

    public boolean AddNewConnection(Node recievingNode, NodeTransferFunction transferFunction)
    {
        boolean doesConnectionExist = connections.stream().anyMatch(connection -> connection.DoesMatchNodes(this, recievingNode));
        if(doesConnectionExist)
        {
            connections.add(new NodeConnection(this, recievingNode, transferFunction));
        }
        return doesConnectionExist;
    }

}
