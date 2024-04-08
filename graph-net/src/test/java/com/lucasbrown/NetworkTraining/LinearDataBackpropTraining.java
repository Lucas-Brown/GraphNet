package com.lucasbrown.NetworkTraining;

import com.lucasbrown.GraphNetwork.Distributions.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Distributions.OpenFilter;
import com.lucasbrown.GraphNetwork.Global.BackpropTrainer;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Nodes.ComplexNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;

public class LinearDataBackpropTraining {
    
    private static int counter = 0;

    private int N = 10;
    private Double[][] inputData;
    private Double[][] outputData;

    private void initializeInputData()
    {
        inputData = new Double[N][1];
        for (int i = 0; i < N; i++) {
            inputData[i] = new Double[]{Double.valueOf(i)};
        }
    } 

    private void initializeOutputData()
    {
        outputData = new Double[N][1];
        for (int i = 0; i < N; i++) {
            outputData[i] = new Double[]{Double.valueOf(i)};
        }
    } 

    public static void main(String[] args){
        LinearDataBackpropTraining linear = new LinearDataBackpropTraining();
        linear.initializeInputData();
        linear.initializeOutputData();

        GraphNetwork net = new GraphNetwork();

        InputNode in = ComplexNode.asInputNode(ActivationFunction.LINEAR);
        //INode hidden = net.createHiddenNode(ActivationFunction.LINEAR);
        OutputNode out = ComplexNode.asOutputNode(ActivationFunction.LINEAR);

        in.setName("Input");
        //hidden.setName("Hidden");
        out.setName("Output");

        net.addNodeToNetwork(in);
        net.addNodeToNetwork(out);

        //net.addNewConnection(in, hidden, new BellCurveDistribution(0, 1));
        //net.addNewConnection(hidden, out, new BellCurveDistribution(-1, 1));
        net.addNewConnection(in, out, new BellCurveDistribution(-1, 1));

        BackpropTrainer bt = new BackpropTrainer(net, new ErrorFunction.MeanSquaredError());

        bt.setTrainingData(linear.inputData, linear.outputData);
        bt.trainNetwork(10000, 100);

        net.deactivateAll();
        net.setInputOperation(nodeMap -> nodeMap.values().iterator().next().acceptUserForwardSignal(linear.inputData[counter++][0]));
        for (int i = 0; i < linear.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
