package com.lucasbrown.InferenceAndAssessment;

import com.lucasbrown.GraphNetwork.Global.DataGraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Local.IInputNode;
import com.lucasbrown.GraphNetwork.Local.Node;

public class dataRepresentationInferenceTest {
    
    public static <T extends Node & IInputNode> void main(String[] args)
    {
        // basic network setup
        DataGraphNetwork net = new DataGraphNetwork();

        T in = net.createInputNode(ActivationFunction.LINEAR);
        Node out = net.createOutputNode(ActivationFunction.LINEAR);
        Node hidden = net.createHiddenNode(ActivationFunction.LINEAR);

        in.setName("Input");
        out.setName("Output");
        hidden.setName("Hidden");

        net.addNewConnection(in, hidden, new BellCurveDistribution(0, 1));
        net.addNewConnection(hidden, out, new BellCurveDistribution(0, 1));

        // just test inference
        net.setInputOperation(input_nodes -> in.acceptUserForwardSignal(1));
        net.setOutputOperation(output_nodes -> System.out.println(net.toString()));

        for(int i = 0; i < 25; i++){
            net.inferenceStep();
        }
    }
}
