package src.Tests;

import java.util.HashSet;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.function.Consumer;

import src.GraphNetwork.Global.GraphNetwork;
import src.GraphNetwork.Local.ActivationFunction;
import src.GraphNetwork.Local.BellCurveDistribution;
import src.GraphNetwork.Local.InputNode;
import src.GraphNetwork.Local.Node;
import src.GraphNetwork.Local.OutputNode;

/**
 * Test for a graph network alternating between no signal and 1
 */
public class SwitchNet
{

    public static void main(String[] args)
    {
        GraphNetwork net = new GraphNetwork();

        InputNode n1 = net.createInputNode(ActivationFunction.SIGMOID);
        OutputNode n2 = net.createOutputNode(ActivationFunction.SIGMOID);

        net.addNewConnection(n1, n2, new BellCurveDistribution(1, 1));
        
        for(int i = 0; i < 100; i++)
        {
            n1.recieveInputSignal(0);

            // Transfer all signals
            net.trainingStep();

            System.out.println(net.allActiveNodesString());

            // Train the network to output a 1 every other step
            n2.correctOutputValue(i % 2 == 0 ? 1.0 : null);

        }

        System.out.println("\nTRAINING STOP\n");
        
        for(int i = 0; i < 100; i++)
        {
            n1.recieveInputSignal(0);
            
            System.out.println(net.allActiveNodesString());
        }

    }

}