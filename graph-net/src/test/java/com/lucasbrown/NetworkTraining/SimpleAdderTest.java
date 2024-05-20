package com.lucasbrown.NetworkTraining;

import java.util.Objects;
import java.util.Random;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Global.ArcBuilder;
import com.lucasbrown.GraphNetwork.Global.BackpropTrainer;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.NodeBuilder;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.NodeBase;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.SimpleNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster;
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
            inputData[i] = new Double[] { doubleOrNothing(), doubleOrNothing(), doubleOrNothing() };
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

        NodeBuilder nodeBuilder = new NodeBuilder(net);

        nodeBuilder.setActivationFunction(ActivationFunction.LINEAR);
        nodeBuilder.setNodeClass(SimpleNode.class);
        nodeBuilder.setOutputDistSupplier(NormalDistribution::getStandardNormalDistribution);
        nodeBuilder.setProbabilityDistSupplier(BetaDistribution::getUniformBetaDistribution);
        nodeBuilder.setAsInputNode();

        InputNode in1 = (InputNode) nodeBuilder.build();
        InputNode in2 = (InputNode) nodeBuilder.build();
        InputNode in3 = (InputNode) nodeBuilder.build();

        nodeBuilder.setAsOutputNode();

        OutputNode out = (OutputNode) nodeBuilder.build();

        in1.setName("Input 1");
        in2.setName("Input 2");
        in3.setName("Input 3");
        out.setName("Output");

        ArcBuilder arcBuilder = new ArcBuilder(net);
        arcBuilder.setFilterSupplier(NormalBetaFilter::getStandardNormalBetaFilter);
        arcBuilder.setFilterAdjusterSupplier(NormalBetaFilterAdjuster::new);

        arcBuilder.build(in1, out);
        arcBuilder.build(in2, out);
        arcBuilder.build(in3, out);

        BackpropTrainer bt = new BackpropTrainer(net, new ErrorFunction.MeanSquaredError());
        bt.epsilon = 1;

        bt.setTrainingData(adder.inputData, adder.outputData);

        bt.trainNetwork(10000, 100);
        net.deactivateAll();
        net.setInputOperation(nodeMap -> BackpropTrainer.applyInputToNode(nodeMap, adder.inputData, counter++));
        for (int i = 0; i < adder.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
