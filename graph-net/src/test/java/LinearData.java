import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Local.InputNode;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.GraphNetwork.Local.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Signal;

public class LinearData {

    private static int index = 0;
    private static double[] inputData = new double[]{0,1,2,3,4,5};
    private static double[] outputData = new double[]{5,4,3,2,1,0};

    private static InputNode in;
    private static OutputNode out;

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

        net.setInputOperation(LinearData::inputOperation);
        net.setOutputOperation(LinearData::trainOutputOperation);

        for (int i = 0; i < 1000; i++) {

            // Transfer all signals
            net.trainingStep();
            System.out.println(net.allActiveNodesString());
        }

        System.out.println("\nTRAINING STOP\n");

        net.deactivateAll();
        net.setOutputOperation(LinearData::readOutputOperation);

        for (int i = 0; i < 10; i++) {
            net.inferenceStep();
        }

    }

    public static void inputOperation() {
        index = (index + 1) % inputData.length;
        in.recieveForwardSignal(new Signal(null, in, inputData[index]));
    }

    public static void trainOutputOperation() {
        out.recieveBackwardSignal(new Signal(out, null, outputData[index]));
    }

    public static void readOutputOperation() {
        System.out.println(out.getValueOrNull());
    }
}
