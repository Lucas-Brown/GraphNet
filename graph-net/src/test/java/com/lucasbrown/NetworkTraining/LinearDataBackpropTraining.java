package com.lucasbrown.NetworkTraining;

import com.lucasbrown.GraphNetwork.Global.ArcBuilder;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.NodeBuilder;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Filters.FlatRateFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.NormalPeakFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.OpenFilter;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.SimpleNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.ComplexCombinator;
import com.lucasbrown.NetworkTraining.BackpropTrainer;
import com.lucasbrown.NetworkTraining.NewtonTrainer;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BernoulliDistribution;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BernoulliDistributionAdjuster;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BetaDistribution;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BetaDistributionAdjuster;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BetaDistributionAdjuster2;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NoAdjustments;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NormalBernoulliFilterAdjuster;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NormalBetaFilterAdjuster;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NormalBetaFilterAdjuster2;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NormalDistribution;
import com.lucasbrown.NetworkTraining.OutputDerivatives.ErrorFunction;

public class LinearDataBackpropTraining {
    
    private static int counter = 0;

    private int N = 25;
    private Double[][] inputData;
    private Double[][] outputData;

    private void initializeInputData()
    {
        inputData = new Double[N][1];
        for (int i = 0; i < N; i++) {
            inputData[i] = new Double[]{Double.valueOf(i)/N};
        }
    } 

    private void initializeOutputData()
    {
        outputData = new Double[N][1];
        for (int i = 0; i < N; i++) {
            outputData[(i + 2) % N] = new Double[]{Double.valueOf(i)};
        }
    } 

    public static void main(String[] args){
        LinearDataBackpropTraining linear = new LinearDataBackpropTraining();
        linear.initializeInputData();
        linear.initializeOutputData();

        GraphNetwork net = new GraphNetwork();

        NodeBuilder nodeBuilder = new NodeBuilder(net);

        nodeBuilder.setActivationFunction(ActivationFunction.LINEAR);
        nodeBuilder.setNodeConstructor(ComplexCombinator::new);
        nodeBuilder.setOutputDistSupplier(NormalDistribution::getStandardNormalDistribution);
        nodeBuilder.setProbabilityDistSupplier(BernoulliDistribution::getEvenDistribution);
        nodeBuilder.setProbabilityDistAdjusterSupplier(BernoulliDistributionAdjuster::new);

        nodeBuilder.setAsInputNode();
        InputNode in = (InputNode) nodeBuilder.build();

        nodeBuilder.setAsHiddenNode();
        ITrainable hidden = (ITrainable) nodeBuilder.build();

        nodeBuilder.setAsOutputNode();
        OutputNode out = (OutputNode) nodeBuilder.build();

        in.setName("Input");
        hidden.setName("Hidden");
        out.setName("Output");

        ArcBuilder arcBuilder = new ArcBuilder(net);
        // arcBuilder.setFilterSupplier(NormalPeakFilter::getStandardNormalBetaFilter);
        // arcBuilder.setFilterAdjusterSupplier(NormalBernoulliFilterAdjuster::new);
        arcBuilder.setFilterSupplier(() -> new FlatRateFilter(0.5));
        arcBuilder.setFilterAdjusterSupplier(NoAdjustments::new);

        arcBuilder.build(in, hidden);
        arcBuilder.build(hidden, out);

        NewtonTrainer bt = new NewtonTrainer(net, new ErrorFunction.MeanSquaredError());
        bt.epsilon = 1;

        bt.setTrainingData(linear.inputData, linear.outputData);

        bt.trainNetwork(10000, 1);
        net.deactivateAll();
        net.setInputOperation(nodeMap -> BackpropTrainer.applyInputToNode(nodeMap, linear.inputData, counter++));
        for (int i = 0; i < linear.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
