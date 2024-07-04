package com.lucasbrown.NetworkTraining;

import java.util.Objects;
import java.util.Random;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Global.Network.ArcBuilder;
import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.Network.NodeBuilder;
import com.lucasbrown.GraphNetwork.Global.Trainers.Trainer;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Nodes.ComplexNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistributionAdjuster2;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster2;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalPeakFilter;

public class AdderTest {

    private static Random rng = new Random();

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
        AdderTest adder = new AdderTest();
        adder.initializeInputData();
        adder.initializeOutputData();

        GraphNetwork net = new GraphNetwork();

        NodeBuilder nodeBuilder = new NodeBuilder(net);

        nodeBuilder.setActivationFunction(ActivationFunction.LINEAR);
        nodeBuilder.setNodeConstructor(ComplexNode::new);
        nodeBuilder.setOutputDistSupplier(NormalDistribution::getStandardNormalDistribution);
        // nodeBuilder.setOutputDistAdjusterSupplier(NormalDistributionFromData::new);
        nodeBuilder.setProbabilityDistSupplier(BetaDistribution::getUniformBetaDistribution);
        nodeBuilder.setProbabilityDistAdjusterSupplier(BetaDistributionAdjuster2::new);

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
        // arcBuilder.setFilterSupplier(OpenFilter::new);
        // arcBuilder.setFilterAdjusterSupplier(NoAdjustments::new);
        arcBuilder.setFilterSupplier(NormalPeakFilter::getStandardNormalBetaFilter);
        arcBuilder.setFilterAdjusterSupplier(NormalBetaFilterAdjuster2::new);

        arcBuilder.build(in1, out);
        arcBuilder.build(in2, out);
        arcBuilder.build(in3, out);


        Trainer trainer = Trainer.getDefaultTrainer(net, adder.inputData, adder.outputData);
        trainer.trainNetwork(10000, 1000);
    }
}
