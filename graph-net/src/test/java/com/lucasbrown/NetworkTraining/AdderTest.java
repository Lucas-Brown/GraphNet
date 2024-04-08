package com.lucasbrown.NetworkTraining;

import java.util.Random;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Distributions.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Distributions.OpenFilter;
import com.lucasbrown.GraphNetwork.Global.BackpropTrainer;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Nodes.ComplexNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;

public class AdderTest {
    
    private static Random rng = new Random();
    private static int counter = 0;

    private int N = 5;
    private Double[][] inputData;
    private Double[][] outputData;

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
            outputData[i] = new Double[]{Stream.of(inputData[i]).mapToDouble(d -> d).sum()};
        }
    } 

    public static void main(String[] args){
        AdderTest adder = new AdderTest();
        adder.initializeInputData();
        adder.initializeOutputData();

        GraphNetwork net = new GraphNetwork();

        InputNode in1 = ComplexNode.asInputNode(ActivationFunction.LINEAR);
        InputNode in2 = ComplexNode.asInputNode(ActivationFunction.LINEAR);
        InputNode in3 = ComplexNode.asInputNode(ActivationFunction.LINEAR);
        OutputNode out = ComplexNode.asOutputNode(ActivationFunction.LINEAR);

        in1.setName("Input 1");
        in2.setName("Input 2");
        in3.setName("Input 3");
        out.setName("Output");

        net.addNodeToNetwork(in1);
        net.addNodeToNetwork(in2);
        net.addNodeToNetwork(in3);
        net.addNodeToNetwork(out);

        net.addNewConnection(in1, out, new BellCurveDistribution(0, 1));
        net.addNewConnection(in2, out, new BellCurveDistribution(0, 1));
        net.addNewConnection(in3, out, new BellCurveDistribution(0, 1));

        BackpropTrainer bt = new BackpropTrainer(net, new ErrorFunction.MeanSquaredError());
        bt.epsilon = 0.05;

        bt.setTrainingData(adder.inputData, adder.outputData);
        bt.trainNetwork(1000, 100);

        net.deactivateAll();
        net.setInputOperation(nodeMap -> BackpropTrainer.applyInputToNode(nodeMap, adder.inputData, counter++));
        for (int i = 0; i < adder.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
