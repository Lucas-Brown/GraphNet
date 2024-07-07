package com.lucasbrown.NetworkTraining;

import java.util.Objects;
import java.util.Random;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Global.ArcBuilder;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.NodeBuilder;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Edge;
import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.NormalPeakFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.OpenFilter;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.NodeBase;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.SimpleNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.ComplexCombinator;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BetaDistribution;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BetaDistributionAdjuster2;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BetaDistributionFromData;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BetaDistributionFromData2;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.IExpectationAdjuster;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.ITrainableDistribution;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NormalBetaFilterAdjuster;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NormalBetaFilterAdjuster2;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NormalDistribution;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NormalDistributionFromData;
import com.lucasbrown.NetworkTraining.OutputDerivatives.ErrorFunction;
import com.lucasbrown.NetworkTraining.Trainers.Trainer;

public class PeakTest {

    private int N = 100;
    private static Double[][] inputData = new Double[][] { new Double[] {-1d}, new Double[] { 0d }, new Double[] { 1d }, new Double[]{ null } };
    private static Double[][] outputData = new Double[][] { new Double[] { null }, new Double[] { null }, new Double[] { 1d }, new Double[] { null } };


    public static void main(String[] args) {
        GraphNetwork net = new GraphNetwork();

        NodeBuilder nodeBuilder = new NodeBuilder(net);

        nodeBuilder.setActivationFunction(ActivationFunction.LINEAR);
        nodeBuilder.setNodeConstructor(ComplexCombinator::new);
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

        Trainer trainer = Trainer.getDefaultTrainer(net, inputData, outputData);

        trainer.trainNetwork(10000, 1000);
    }
}
