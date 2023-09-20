package src.Tests;

import java.util.HashSet;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.function.Consumer;

import src.GraphNetwork.Global.GraphNetwork;
import src.GraphNetwork.Local.BellCurveDistribution;
import src.GraphNetwork.Local.Node;

/**
 * Test for a graph network alternating between no signal and 1
 */
public class SwitchNet
{

    public static void main(String[] args)
    {
        GraphNetwork net = new GraphNetwork();

        Node n1 = net.CreateNewNode();
        Node n2 = net.CreateNewNode();

        net.AddNewConnection(n1, n2, new BellCurveDistribution(0f, 1f, 0.5f));
        net.AddNewConnection(n2, n1, new BellCurveDistribution(0f, 1f, 0.5f));
        
        boolean state = false;
        for(int i = 0; i < 10000; i++)
        {
            // Transfer all signals
            net.RecieveSignals();

            // Train the network to output a 1 every other step
            if(state = !state)
            {
                net.CorrectNodeValue(n1, 1);
            }

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