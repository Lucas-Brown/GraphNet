package src.Tests;

import src.GraphNetwork.Global.GraphNetwork;
import src.GraphNetwork.Local.ActivationFunction;
import src.GraphNetwork.Local.BellCurveDistribution;
import src.GraphNetwork.Local.InputNode;
import src.GraphNetwork.Local.Node;
import src.GraphNetwork.Local.OutputNode;

/**
 * Test for a graph network alternating between no signal and 1
 */
public class SwitchNet {

    private static boolean alternating = false;
    private static InputNode in;
    private static OutputNode out;

    private static final double tollerance = 0.01;
    private static int count = 0;

    public static void main(String[] args) {
        GraphNetwork net = new GraphNetwork();

        in = net.createInputNode(ActivationFunction.SIGMOID);
        out = net.createOutputNode(ActivationFunction.SIGMOID);
        Node hidden = net.createHiddenNode(ActivationFunction.SIGMOID);

        in.setName("Input");
        out.setName("Output");
        hidden.setName("Hidden");

        net.addNewConnection(in, hidden, new BellCurveDistribution(1, 2, 10000));
        net.addNewConnection(hidden, out, new BellCurveDistribution(1, 2, 10000));

        net.setInputOperation(SwitchNet::inputOperation);
        net.setOutputOperation(SwitchNet::outputOperation);


        for (int i = 0; i < 100; i++) {

            // Transfer all signals
            net.trainingStep();

            //if((i % 1000) == 0 || (i % 1000) == 1)
                System.out.println(net.allActiveNodesString());


        }

        System.out.println("\nTRAINING STOP\n");

        net.setOutputOperation(SwitchNet::scoringOperation);

        for (int i = 0; i < 10000; i++) {
            net.step();

            System.out.println(net.allActiveNodesString());
        }

        System.out.println("Score = " + count);

    }

    public static void inputOperation()
    {
        in.recieveInputSignal((alternating = !alternating) ? 0 : 1);
    }

    public static void outputOperation()
    {
        out.correctOutputValue(alternating ? 0.1 : 0.9);
    }

    public static void scoringOperation()
    {
        if(alternating)
        {
            if(!out.isActive())
            {
                count++;
            }
        }
        else
        {
            if(out.isActive() && Math.abs(out.getValue() + 1) < tollerance)
            {
                count++;
            }
        }
            
    }

}