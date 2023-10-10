package src.Tests;

import src.GraphNetwork.Global.GraphNetwork;
import src.GraphNetwork.Local.ActivationFunction;
import src.GraphNetwork.Local.InputNode;
import src.GraphNetwork.Local.Node;
import src.GraphNetwork.Local.OutputNode;
import src.GraphNetwork.Local.StaticUniformDistribution;

public class FeedForward {
    
    private static boolean alternating = false;
    private static InputNode in;
    private static OutputNode out;

    public static void main(String[] args) {
        GraphNetwork net = new GraphNetwork();

        in = net.createInputNode(ActivationFunction.SIGMOID);
        out = net.createOutputNode(ActivationFunction.SIGMOID);
        Node hidden = net.createHiddenNode(ActivationFunction.SIGMOID);

        in.setName("Input");
        out.setName("Output");
        hidden.setName("Hidden");

        net.addNewConnection(in, hidden, new StaticUniformDistribution());
        net.addNewConnection(hidden, out, new StaticUniformDistribution());

        net.setInputOperation(FeedForward::inputOperation);
        net.setOutputOperation(FeedForward::outputOperation);


        for (int i = 0; i < 100000; i++) {

            // Transfer all signals
            net.trainingStep();

            if((i % 1000) == 0 || (i % 1000) == 1)
                System.out.println(net.allActiveNodesString());


        }

        System.out.println("\nTRAINING STOP\n");

    }

    
    public static void inputOperation()
    {
        in.recieveInputSignal((alternating = !alternating) ? 0 : 1);
    }

    public static void outputOperation()
    {
        out.correctOutputValue(!alternating ? 0.1 : 0.9);
    }
}
