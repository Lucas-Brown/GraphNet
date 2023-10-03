package src.Tests;

import java.util.Random;

import src.GraphNetwork.Global.GraphNetwork;
import src.GraphNetwork.Local.ActivationFunction;
import src.GraphNetwork.Local.BellCurveDistribution;
import src.GraphNetwork.Local.InputNode;
import src.GraphNetwork.Local.Node;
import src.GraphNetwork.Local.OutputNode;

/**
 * Test for a graph network alternating between 0 and 1 
 */
public class Alternating
{
    private static GraphNetwork net;
    private static InputNode n1;
    private static OutputNode n2;
    
    private static boolean waitingFor0 = true;
    private static final double tollerance = 0.01;
    private static int count = 0;

    public static void main(String[] args) {
        Random rand = new Random();
        net = new GraphNetwork();

        n1 = net.createInputNode(ActivationFunction.SIGMOID);
        n2 = net.createOutputNode(ActivationFunction.SIGMOID);
        Node h1 = net.createHiddenNode(ActivationFunction.SIGMOID);
        Node h2 = net.createHiddenNode(ActivationFunction.SIGMOID);

        n1.setName("Input");
        n2.setName("Output");
        h1.setName("Hidden 1");
        h2.setName("Hidden 2");

        net.addNewConnection(n1, h1, new BellCurveDistribution(rand.nextDouble(), 1, 1000));
        net.addNewConnection(n1, h2, new BellCurveDistribution(rand.nextDouble(), 1, 1000));
        net.addNewConnection(h1, h2, new BellCurveDistribution(rand.nextDouble(), 1, 1000));
        net.addNewConnection(h2, h1, new BellCurveDistribution(rand.nextDouble(), 1, 1000));
        net.addNewConnection(h1, n2, new BellCurveDistribution(rand.nextDouble(), 1, 1000));
        net.addNewConnection(h2, n2, new BellCurveDistribution(rand.nextDouble(), 1, 1000));

        net.setInputOperation(Alternating::inputOperation);
        net.setOutputOperation(Alternating::outputOperation);


        for (int i = 0; i < 1000000; i++) {

            // Transfer all signals
            net.trainingStep();

            //System.out.println(net.allActiveNodesString());


        }

        System.out.println("\nTRAINING STOP\n");

        //net.setOutputOperation(Alternating::scoringOperation);
        net.setOutputOperation(null);

        for (int i = 0; i < 10000; i++) {
            net.step();

            System.out.println(net.allActiveNodesString());
        }

        System.out.println("Score = " + count);

    }

    public static void inputOperation()
    {
        if(net.isNetworkDead())
        {
            n1.recieveInputSignal(0);
        }
    }

    public static void outputOperation()
    {
        if(n2.isActive())
        {
            n2.correctOutputValue((waitingFor0 = !waitingFor0) ? 0.0 : 1.0);
        }
        
    }

    /* 
    public static void scoringOperation()
    {
        if(alternating)
        {
            if(n2.isActive() && Math.abs(n2.getValue() - 1) < tollerance)
            {
                count++;
            }
        }
        else
        {
            if(!n2.isActive())
            {
                count++;
            }
        }
            
    }*/

}