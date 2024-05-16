package com.lucasbrown.NetworkTraining;

import java.util.Objects;
import java.util.Random;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Global.BackpropTrainer;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.SimpleNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster2;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistribution;

public class SimpleAdderTest {

    private static Random rng = new Random();
    private static int counter = 0;

    private int N = 100;
    private Double[][] inputData = new Double[][] { new Double[] { 1d, null, null }, new Double[] { null, null, null },
            new Double[] { 2d, 1d, 3d } };
    private Double[][] outputData = new Double[][] { new Double[] { 6d }, new Double[] { 1d }, new Double[] { null } };

    private void initializeInputData() {
        inputData = new Double[N][1];
        for (int i = 0; i < N; i++) {
            inputData[i] = new Double[] { doubleOrNothing(), doubleOrNothing() };
            // if(inputData[i][0] == null & inputData[i][1] == null & inputData[i][2] ==
            // null){
            // i--;
            // }
        }
    }

    private void initializeOutputData() {
        outputData = new Double[N][1];
        for (int i = 0; i < N; i++) {
            Double[] data = inputData[i];
            if (Stream.of(data).allMatch(Objects::isNull)) {
                outputData[(i + 1) % N] = new Double[] { null };
            } else {
                outputData[(i + 1) % N] = new Double[] {
                        Stream.of(inputData[i]).filter(Objects::nonNull).mapToDouble(d -> d).sum() };
            }

        }
    }

    private Double doubleOrNothing() {
        return rng.nextBoolean() ? rng.nextGaussian() : null;
    }

    public static void main(String[] args) {
        SimpleAdderTest adder = new SimpleAdderTest();
        adder.initializeInputData();
        adder.initializeOutputData();

        GraphNetwork net = new GraphNetwork();

        SimpleNode s_in_1 = new SimpleNode(net, ActivationFunction.LINEAR, new NormalDistribution(0, 1),
                new BetaDistribution(1, 1, 10));
        SimpleNode s_in_2 = new SimpleNode(net, ActivationFunction.LINEAR, new NormalDistribution(0.5, 1),
                new BetaDistribution(1, 1, 10));

        InputNode in1 = new InputNode(s_in_1);
        InputNode in2 = new InputNode(s_in_2);

        SimpleNode s_out = new SimpleNode(net, ActivationFunction.LINEAR, new NormalDistribution(0, 1),
                new BetaDistribution(1, 1, 10));
        OutputNode out = new OutputNode(s_out);

        in1.setName("Input 1");
        in2.setName("Input 2");
        // in3.setName("Input 3");
        out.setName("Output");

        net.addNodeToNetwork(in1);
        net.addNodeToNetwork(in2);
        // net.addNodeToNetwork(in3);
        net.addNodeToNetwork(out);

        NormalBetaFilter b1 = new NormalBetaFilter(0, 1);
        NormalBetaFilter b2 = new NormalBetaFilter(0, 1);
        Arc a1 = net.addNewConnection(in1, out, b1, new NormalBetaFilterAdjuster2(b1, (NormalDistribution) in1.getOutputDistribution(), (BetaDistribution) in1.getSignalChanceDistribution()));
        Arc a2 = net.addNewConnection(in2, out, b2, new NormalBetaFilterAdjuster2(b2, (NormalDistribution) in2.getOutputDistribution(), (BetaDistribution) in2.getSignalChanceDistribution()));
        // Arc a3 = net.addNewConnection(in3, out, new BellCurveDistribution(0, 1, 10,
        // 100));

        // net.addNewConnection(in1, out, new OpenFilter());
        // net.addNewConnection(in2, out, new OpenFilter());
        // net.addNewConnection(in3, out, new OpenFilter());

        BackpropTrainer bt = new BackpropTrainer(net, new ErrorFunction.MeanSquaredError());
        bt.epsilon = 0.01;

        bt.setTrainingData(adder.inputData, adder.outputData);

        bt.trainNetwork(10000, 5);
        net.deactivateAll();
        net.setInputOperation(nodeMap -> BackpropTrainer.applyInputToNode(nodeMap, adder.inputData, counter++));
        for (int i = 0; i < adder.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
