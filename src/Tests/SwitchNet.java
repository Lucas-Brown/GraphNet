package src.Tests;

import src.GraphNetwork.Global.GraphNetwork;
import src.GraphNetwork.Local.ActivationFunction;
import src.GraphNetwork.Local.BellCurveDistribution;
import src.GraphNetwork.Local.InputNode;
import src.GraphNetwork.Local.OutputNode;

/**
 * Test for a graph network alternating between no signal and 1
 */
public class SwitchNet {

    private static boolean alternating = false;
    private static InputNode n1;
    private static OutputNode n2;

    private static final double tollerance = 0.01;
    private static int count = 0;

    public static void main(String[] args) {
        GraphNetwork net = new GraphNetwork();

        n1 = net.createInputNode(ActivationFunction.SIGMOID);
        n2 = net.createOutputNode(ActivationFunction.SIGMOID);
        net.addNewConnection(n1, n2, new BellCurveDistribution(1.2, 1, 1000));

        net.setInputOperation(SwitchNet::inputOperation);
        net.setOutputOperation(SwitchNet::outputOperation);


        for (int i = 0; i < 100000; i++) {

            // Transfer all signals
            net.trainingStep();

            //System.out.println(net.allActiveNodesString());


        }

        System.out.println("\nTRAINING STOP\n");

        net.setOutputOperation(SwitchNet::scoringOperation);

        for (int i = 0; i < 10000; i++) {
            net.step();

            //System.out.println(net.allActiveNodesString());
        }

        System.out.println("Score = " + count);

    }

    public static void inputOperation()
    {
        n1.recieveInputSignal((alternating = !alternating) ? 0 : 1);
    }

    public static void outputOperation()
    {
        n2.correctOutputValue(alternating ? 1.0 : null);
    }

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
            
    }

}