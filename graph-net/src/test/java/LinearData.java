import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.GraphNetwork.Local.Signal;
import com.lucasbrown.GraphNetwork.Local.ReferenceStructure.InputReferenceNode;
import com.lucasbrown.GraphNetwork.Local.ReferenceStructure.OutputReferenceNode;

public class LinearData {

    private static int index = 1;
    private static double[] inputData = new double[]{0,1};
    private static double[] outputData = new double[]{0,2};

    private static InputReferenceNode in;
    private static OutputReferenceNode out;

    public static void main(String[] args) {
        GraphNetwork net = new GraphNetwork();

        in = net.createInputNode(ActivationFunction.LINEAR);
        out = net.createOutputNode(ActivationFunction.LINEAR);
        Node hidden = net.createHiddenNode(ActivationFunction.LINEAR);

        in.setName("Input");
        out.setName("Output");
        hidden.setName("Hidden1");

        net.addNewConnection(in, hidden, new BellCurveDistribution(0, 1));
        net.addNewConnection(hidden, out, new BellCurveDistribution(0, 1));

        net.setInputOperation(LinearData::inputOperation);
        net.setOutputOperation(LinearData::trainOutputOperation);

        for (int i = 0; i < 10000; i++) {

            // Transfer all signals
            net.trainingStep();
            System.out.println(net.allActiveNodesString());
        }

        System.out.println("\nTRAINING STOP\n");

        net.deactivateAll();
        net.setOutputOperation(LinearData::readOutputOperation);

        for (int i = 0; i < 100; i++) {
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
