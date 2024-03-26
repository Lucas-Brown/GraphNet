import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Global.ReferenceGraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.GraphNetwork.Local.ReferenceStructure.InputReferenceNode;
import com.lucasbrown.GraphNetwork.Local.ReferenceStructure.OutputReferenceNode;

public class LinearData {

    private static ReferenceGraphNetwork net;

    private static double[] inputData = new double[]{0,1,2,3,4,5,6,7,8,9};
    private static double[] outputData = new double[]{1,3,5,7,9,11,13,15,17,19};
    private static int index = inputData.length-1;

    private static InputReferenceNode in;
    private static OutputReferenceNode out;

    public static void main(String[] args) {
        net = new ReferenceGraphNetwork();

        in = net.createInputNode(ActivationFunction.LINEAR);
        out = net.createOutputNode(ActivationFunction.LINEAR);
        Node hidden = net.createHiddenNode(ActivationFunction.LINEAR);

        in.setName("Input");
        out.setName("Output");
        hidden.setName("Hidden1");

        net.addNewConnection(in, hidden, new BellCurveDistribution(0, 1));
        net.addNewConnection(hidden, hidden, new BellCurveDistribution(0, 1));
        net.addNewConnection(hidden, out, new BellCurveDistribution(0, 1));

        //net.addNewConnection(in, out, new BellCurveDistribution(0, 1));

        net.setInputOperation(LinearData::inputOperation);
        net.setOutputOperation(LinearData::trainOutputOperation);

        for (int i = 0; i < 1000; i++) {

            // Transfer all signals
            net.trainingStep();
            System.out.println(net);
        }

        System.out.println("\nTRAINING STOP\n");

        index = inputData.length-1;
        net.setInputOperation(LinearData::inferenceOperation);
        net.setOutputOperation(LinearData::readOutputOperation);
        net.deactivateAll();

        for(int trial = 0; trial < 10; trial++)
        {
            for (int i = 0; i < inputData.length; i++) {
                net.inferenceStep();
            }
        }

    }

    public static void inputOperation(HashMap<Integer, InputReferenceNode> inputMap) {
        if((index = (index + 1) % inputData.length) == 0)
        {
            net.deactivateAll();
        };
        in.acceptUserForwardSignal(inputData[index]);
    }

    public static void trainOutputOperation(HashMap<Integer, OutputReferenceNode> outputMap) {
        out.acceptUserBackwardSignal(outputData[index]);
    }

    public static void inferenceOperation(HashMap<Integer, InputReferenceNode> inputMap) {
        if((index = (index + 1) % inputData.length) == 0)
        {
            net.deactivateAll();
        };
        in.acceptUserInferenceSignal(inputData[index]);
    }

    public static void readOutputOperation(HashMap<Integer, OutputReferenceNode> outputMap) {
        System.out.println(out.getValueOrNull());
    }
}
