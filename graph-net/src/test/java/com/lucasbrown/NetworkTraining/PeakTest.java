package com.lucasbrown.NetworkTraining;

import java.util.Objects;
import java.util.Random;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Global.Network.ArcBuilder;
import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.Network.NodeBuilder;
import com.lucasbrown.GraphNetwork.Global.Trainers.BackpropTrainer;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Nodes.ComplexNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.NodeBase;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.SimpleNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistributionAdjuster2;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistributionFromData;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistributionFromData2;
import com.lucasbrown.NetworkTraining.DataSetTraining.IExpectationAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.IFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.ITrainableDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalPeakFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster2;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistributionFromData;
import com.lucasbrown.NetworkTraining.DataSetTraining.OpenFilter;

public class PeakTest {

    private static Random rng = new Random();
    private static int counter = 0;

    private int N = 100;
    private static Double[][] inputData = new Double[][] { new Double[] {-1d}, new Double[] { 0d }, new Double[] { 1d }, new Double[]{ null } };
    private static Double[][] outputData = new Double[][] { new Double[] { null }, new Double[] { null }, new Double[] { 1d }, new Double[] { null } };


    public static void main(String[] args) {
        GraphNetwork net = new GraphNetwork();

        NodeBuilder nodeBuilder = new NodeBuilder(net);

        nodeBuilder.setActivationFunction(ActivationFunction.LINEAR);
        nodeBuilder.setNodeConstructor(ComplexNode::new);
        nodeBuilder.setOutputDistSupplier(NormalDistribution::getStandardNormalDistribution);
        // nodeBuilder.setOutputDistAdjusterSupplier(NormalDistributionFromData::new);
        nodeBuilder.setProbabilityDistSupplier(BetaDistribution::getUniformBetaDistribution);
        nodeBuilder.setProbabilityDistAdjusterSupplier(BetaDistributionAdjuster2::new);

        nodeBuilder.setAsInputNode();

        InputNode in = (InputNode) nodeBuilder.build();

        nodeBuilder.setAsOutputNode();

        OutputNode out = (OutputNode) nodeBuilder.build();

        in.setName("Input");
        out.setName("Output");

        ArcBuilder arcBuilder = new ArcBuilder(net);
        arcBuilder.setFilterSupplier(NormalPeakFilter::getStandardNormalBetaFilter);
        arcBuilder.setFilterAdjusterSupplier(NormalBetaFilterAdjuster2::new);

        arcBuilder.build(in, out);

        BackpropTrainer bt = new BackpropTrainer(net, new ErrorFunction.MeanSquaredError(), false);
        bt.epsilon = 1;

        bt.setTrainingData(inputData, outputData);

        bt.trainNetwork(10000, 100);
        net.deactivateAll();
        net.setInputOperation(nodeMap -> BackpropTrainer.applyInputToNode(nodeMap, inputData, counter++));
        for (int i = 0; i < inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
