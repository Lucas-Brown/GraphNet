package com.lucasbrown.InferenceAndAssessment;

import com.lucasbrown.GraphNetwork.Distributions.BellCurveFilter;
import com.lucasbrown.GraphNetwork.Global.DataGraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.IInputNode;
import com.lucasbrown.GraphNetwork.Local.INode;
import com.lucasbrown.GraphNetwork.Local.DataStructure.InputDataNode;
import com.lucasbrown.GraphNetwork.Local.DataStructure.OutputDataNode;

public class DataRepresentationInferenceTest {
    
    public static void main(String[] args)
    {
        // basic network setup
        DataGraphNetwork net = new DataGraphNetwork();

        InputDataNode in = net.createInputNode(ActivationFunction.LINEAR);
        OutputDataNode out = net.createOutputNode(ActivationFunction.LINEAR);
        INode hidden = net.createHiddenNode(ActivationFunction.LINEAR);

        in.setName("Input");
        out.setName("Output");
        hidden.setName("Hidden");

        net.addNewConnection(in, hidden, new BellCurveFilter(0, 1));
        net.addNewConnection(hidden, out, new BellCurveFilter(0, 1));

        // just test inference
        net.setInputOperation(input_nodes -> in.acceptUserForwardSignal(1));
        net.setOutputOperation(output_nodes -> System.out.println(net.toString()));

        for(int i = 0; i < 25; i++){
            net.inferenceStep();
        }
    }
}
