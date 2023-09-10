package src.Tests;

import src.GraphNetwork.GraphNetwork;
import src.GraphNetwork.Node;
import src.GraphNetwork.NormalTransferFunction;

/**
 * Test for a graph network alternating between 0 and 1 
 */
public class SwitchNet
{
    public static void main(String[] args)
    {
        GraphNetwork net = new GraphNetwork();

        Node n1 = new Node();
        Node n2 = new Node();
        net.nodes.add(n1);
        net.nodes.add(n2);

        n1.AddNewConnection(n2, new NormalTransferFunction(0f, 1f, 0.5f));
        n2.AddNewConnection(n1, new NormalTransferFunction(0f, 1f, 0.5f));
    }
}