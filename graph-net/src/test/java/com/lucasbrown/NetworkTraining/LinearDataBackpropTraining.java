package com.lucasbrown.NetworkTraining;

import com.lucasbrown.GraphNetwork.Distributions.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Distributions.OpenFilter;
import com.lucasbrown.GraphNetwork.Global.BackpropTrainer;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Node;
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

        Node in = net.createInputNode(ActivationFunction.LINEAR);
        //Node hidden = net.createHiddenNode(ActivationFunction.LINEAR);
        Node out = net.createOutputNode(ActivationFunction.LINEAR);

        in.setName("Input");
        //hidden.setName("Hidden");
        out.setName("Output");

        //net.addNewConnection(in, hidden, new BellCurveDistribution(0, 1));
        //net.addNewConnection(hidden, out, new BellCurveDistribution(-1, 1));
        net.addNewConnection(in, out, new BellCurveDistribution(-1, 1));

        BackpropTrainer bt = new BackpropTrainer(net, new ErrorFunction.MeanSquaredError());

        bt.setTrainingData(linear.inputData, linear.outputData);
        bt.trainNetwork(1000, 100);

        net.deactivateAll();
        net.setInputOperation(nodeMap -> nodeMap.values().iterator().next().acceptUserForwardSignal(linear.inputData[counter++][0]));
        for (int i = 0; i < linear.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
