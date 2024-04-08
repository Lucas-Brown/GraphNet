package com.lucasbrown.NetworkTraining;

import java.util.Random;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Distributions.OpenFilter;
import com.lucasbrown.GraphNetwork.Global.BackpropTrainer;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.SimpleNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;

public class AdderTest {
    
    private static Random rng = new Random();
    private static int counter = 0;

    private int N = 15;
    private Double[][] inputData = new Double[][]{new Double[]{1d,2d,3d}, new Double[]{2d,1d,3d}, new Double[]{3d,2d,1d}, new Double[]{3d,2d,1d}};
    private Double[][] outputData = new Double[][]{new Double[]{6d}, new Double[]{6d}, new Double[]{6d}, new Double[]{6d}, new Double[]{6d}};

    private void initializeInputData()
    {
        inputData = new Double[N][1];
        for (int i = 0; i < N; i++) {
            inputData[i] = new Double[]{rng.nextDouble(), rng.nextDouble(), rng.nextDouble()};
        }
    } 

    private void initializeOutputData()
    {
        outputData = new Double[N][1];
        for (int i = 0; i < N; i++) {
            outputData[(i + 1) % N] = new Double[]{Stream.of(inputData[i]).mapToDouble(d -> d).sum()};
        }
    } 

    public static void main(String[] args){
        AdderTest adder = new AdderTest();
        adder.initializeInputData();
        adder.initializeOutputData();

        GraphNetwork net = new GraphNetwork();

        InputNode in1 = SimpleNode.asInputNode(ActivationFunction.LINEAR);
        InputNode in2 = SimpleNode.asInputNode(ActivationFunction.LINEAR);
        InputNode in3 = SimpleNode.asInputNode(ActivationFunction.LINEAR);
        OutputNode out = SimpleNode.asOutputNode(ActivationFunction.LINEAR);

        in1.setName("Input 1");
        in2.setName("Input 2");
        in3.setName("Input 3");
        out.setName("Output");

        net.addNodeToNetwork(in1);
        net.addNodeToNetwork(in2);
        net.addNodeToNetwork(in3);
        net.addNodeToNetwork(out);

        // net.addNewConnection(in1, out, new BellCurveDistribution(0, 1));
        // net.addNewConnection(in2, out, new BellCurveDistribution(0, 1));
        // net.addNewConnection(in3, out, new BellCurveDistribution(0, 1));

        net.addNewConnection(in1, out, new OpenFilter());
        net.addNewConnection(in2, out, new OpenFilter());
        net.addNewConnection(in3, out, new OpenFilter());

        BackpropTrainer bt = new BackpropTrainer(net, new ErrorFunction.MeanSquaredError());
        bt.epsilon = 0.05;

        bt.setTrainingData(adder.inputData, adder.outputData);
        bt.trainNetwork(10000, 1000);

        net.deactivateAll();
        net.setInputOperation(nodeMap -> BackpropTrainer.applyInputToNode(nodeMap, adder.inputData, counter++));
        for (int i = 0; i < adder.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
