package com.lucasbrown.NetworkTraining;

import com.lucasbrown.GraphNetwork.Global.Network.ArcBuilder;
import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.Network.NodeBuilder;
import com.lucasbrown.GraphNetwork.Global.Trainers.BackpropTrainer;
import com.lucasbrown.GraphNetwork.Global.Trainers.NewtonTrainer;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Nodes.ComplexNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.SimpleNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;
import com.lucasbrown.NetworkTraining.DataSetTraining.BernoulliDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.BernoulliDistributionAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistributionAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistributionAdjuster2;
import com.lucasbrown.NetworkTraining.DataSetTraining.FlatRateFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.NoAdjustments;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBernoulliFilterAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalPeakFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.OpenFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster2;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistribution;

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
        nodeBuilder.setNodeConstructor(ComplexNode::new);
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
