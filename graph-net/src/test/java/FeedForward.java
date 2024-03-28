import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Local.InputNode;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.GraphNetwork.Local.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Signal;

public class FeedForward {

    private static InputNode in;
    private static OutputNode out;

    public static void main(String[] args) {
        GraphNetwork net = new GraphNetwork();

        in = net.createInputNode(ActivationFunction.LINEAR);
        out = net.createOutputNode(ActivationFunction.LINEAR);
        Node hidden1 = net.createHiddenNode(ActivationFunction.LINEAR);
        Node hidden2 = net.createHiddenNode(ActivationFunction.LINEAR);

        in.setName("Input");
        out.setName("Output");
        hidden1.setName("Hidden1");
        hidden2.setName("Hidden2");

        net.addNewConnection(in, hidden1, new BellCurveDistribution(0, 1));
        net.addNewConnection(in, hidden2, new BellCurveDistribution(1, 1));
        net.addNewConnection(hidden1, out, new BellCurveDistribution(-1, 1));
        net.addNewConnection(hidden2, out, new BellCurveDistribution(0, 1));

        //net.addNewConnection(in, out, new BellCurveDistribution(0, 1));

        net.setInputOperation(FeedForward::inputOperation);
        net.setOutputOperation(FeedForward::trainOutputOperation);

        for (int i = 0; i < 3; i++) {

            // Transfer all signals
            net.trainingStep();
            System.out.println(net);
        }

        System.out.println("\nTRAINING STOP\n");

        net.deactivateAll();
        net.setOutputOperation(FeedForward::readOutputOperation);

        for (int i = 0; i < 100; i++) {
            net.inferenceStep();
            //System.out.println(net.allActiveNodesString());
        }

    }

    public static void inputOperation(HashMap<Integer, InputNode> inputNodeMap) {
        in.acceptUserForwardSignal(0);
    }

    public static void trainOutputOperation(HashMap<Integer, OutputNode> outputNodeMap) {
        out.acceptUserBackwardSignal(5);
    }

    public static void readOutputOperation(HashMap<Integer, OutputNode> outputNodeMap) {
        System.out.println(out.getValueOrNull());
    }
}
