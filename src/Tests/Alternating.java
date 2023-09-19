package src.Tests;

import java.util.HashSet;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.function.Consumer;

import src.GraphNetwork.GraphNetwork;
import src.GraphNetwork.Node;
import src.GraphNetwork.BellCurveDistribution;

/**
 * Test for a graph network alternating between 0 and 1 
 */
public class Alternating
{

    public static void main(String[] args)
    {
        GraphNetwork net = new GraphNetwork();

        Node n1 = net.CreateNewNode();
        
        net.AddNewConnection(n1, n1, new BellCurveDistribution(0.1f, 1f, 0.9f));
        net.AddNewConnection(n1, n1, new BellCurveDistribution(0.9f, 1f, 0.1f));

        /* 
        Node n2 = new Node();
        Node n3 = new Node();
        net.nodes.add(n1);
        net.nodes.add(n2);
        net.nodes.add(n3);

        net.AddNewConnection(n1, n2, new NormalTransferFunction(0.1f, 1f, 0.9f));
        net.AddNewConnection(n2, n1, new NormalTransferFunction(0.2f, 1f, 0.8f));
        net.AddNewConnection(n1, n3, new NormalTransferFunction(0.3f, 1f, 0.7f));
        net.AddNewConnection(n3, n1, new NormalTransferFunction(0.4f, 1f, 0.6f));
        net.AddNewConnection(n2, n3, new NormalTransferFunction(0.5f, 1f, 0.5f));
        net.AddNewConnection(n3, n2, new NormalTransferFunction(0.6f, 1f, 0.4f));
        */
        
        boolean state = false;
        for(int i = 0; i < 100000; i++)
        {
            // Transfer all signals
            net.RecieveSignals();

            // Train the network to output alternating 0 and 1
            net.CorrectNodeValue(n1, (state = !state) ? 1 : 0);

            net.TransmitSignals();
            net.ReinforceSignals();
            net.PropagateErrors();

        }

        System.out.println("\nTRAINING STOP\n");
        
        int post_fire_count = 0;
        for(int i = 0; i < 1000; i++)
        {
            net.RecieveSignals();
            
            String netStr = net.AllActiveNodesString();
            if(!netStr.trim().isEmpty())
            {
                System.out.println(netStr);
                post_fire_count++;
            }

            net.TransmitSignals();
        }

        System.out.println("steps before auto-stop: " + post_fire_count);
    }

}