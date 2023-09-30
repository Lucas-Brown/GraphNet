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

    public static void main(String[] args) {
        GraphNetwork net = new GraphNetwork();

        n1 = net.createInputNode(ActivationFunction.SIGMOID);
        n2 = net.createOutputNode(ActivationFunction.SIGMOID);
        net.addNewConnection(n1, n2, new BellCurveDistribution(1, 1));

        net.setInputOperation(SwitchNet::inputOperation);
        net.setOutputOperation(SwitchNet::outputOperation);


        for (int i = 0; i < 100; i++) {

            // Transfer all signals
            net.trainingStep();

            System.out.println(net.allActiveNodesString());


        }

        System.out.println("\nTRAINING STOP\n");

        for (int i = 0; i < 100; i++) {
            n1.recieveInputSignal(0);

            System.out.println(net.allActiveNodesString());
        }

    }

    public static void inputOperation()
    {
        n1.recieveInputSignal(0);
    }

    public static void outputOperation()
    {
        n2.correctOutputValue((alternating = !alternating) ? 1.0 : null);
    }

}