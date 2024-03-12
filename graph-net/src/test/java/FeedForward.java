import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.GraphNetwork.Local.Signal;
import com.lucasbrown.GraphNetwork.Local.ReferenceStructure.InputReferenceNode;
import com.lucasbrown.GraphNetwork.Local.ReferenceStructure.OutputReferenceNode;

public class FeedForward {

    private static InputReferenceNode in;
    private static OutputReferenceNode out;

    public static void main(String[] args) {
        GraphNetwork net = new GraphNetwork();

        in = net.createInputNode(ActivationFunction.SIGNED_QUADRATIC);
        out = net.createOutputNode(ActivationFunction.SIGNED_QUADRATIC);
        Node hidden = net.createHiddenNode(ActivationFunction.SIGNED_QUADRATIC);

        in.setName("Input");
        out.setName("Output");
        hidden.setName("Hidden");

        net.addNewConnection(in, hidden, new BellCurveDistribution(0, 1));
        net.addNewConnection(hidden, out, new BellCurveDistribution(0, 1));

        net.setInputOperation(FeedForward::inputOperation);
        net.setOutputOperation(FeedForward::trainOutputOperation);

        for (int i = 0; i < 10000; i++) {

            // Transfer all signals
            net.trainingStep();
            //System.out.println(net.allActiveNodesString());
        }

        System.out.println("\nTRAINING STOP\n");

        net.deactivateAll();
        net.setOutputOperation(FeedForward::readOutputOperation);

        for (int i = 0; i < 100; i++) {
            net.inferenceStep();
        }

    }

    public static void inputOperation() {
        in.recieveForwardSignal(new Signal(null, in, 0));
    }

    public static void trainOutputOperation() {
        out.recieveBackwardSignal(new Signal(out, null, 5));
    }

    public static void readOutputOperation() {
        System.out.println(out.getValueOrNull());
    }
}
