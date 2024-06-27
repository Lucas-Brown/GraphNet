package com.lucasbrown.NetworkTraining;

import java.util.Random;
import com.lucasbrown.GraphNetwork.Global.Network.ArcBuilder;
import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.Network.NodeBuilder;
import com.lucasbrown.GraphNetwork.Global.Trainers.ADAMTrainer;
import com.lucasbrown.GraphNetwork.Global.Trainers.BackpropTrainer;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Nodes.ComplexNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;
import com.lucasbrown.NetworkTraining.DataSetTraining.BernoulliDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.BernoulliDistributionAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistributionAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistributionAdjuster2;
import com.lucasbrown.NetworkTraining.DataSetTraining.NoAdjustments;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBernoulliFilterAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalPeakFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.OpenFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster2;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistribution;

public class MultiLayerPeakTest {

    private static Random rng = new Random();
    private static int counter = 0;
    private static int n_depth = 3;
    private static int N = 100;

    private Double[][] inputData = new Double[][] {
            new Double[] { -2d }, new Double[] { -1d }, new Double[] { -0.5d }, new Double[] { 0d },
            new Double[] { 0.5d }, new Double[] { 1d }, new Double[] { 2d }, new Double[] { null },
            new Double[] { null }, new Double[] { null } };
    private Double[][] outputData = new Double[][] {
            new Double[] { null }, new Double[] { null }, new Double[] { null }, new Double[] { null },
            new Double[] { null }, new Double[] { 0.5d }, new Double[] { 1d }, new Double[] { 1.5d },
            new Double[] { null }, new Double[] { null } };

    private MultiLayerPeakTest(){
        initializeInputData();
        initializeOutputData();
    }


    private void initializeInputData() {
        inputData = new Double[N][1];
        int i;
        for(i = 0; i < N-n_depth; i++){
            inputData[i] = new Double[]{ rng.nextGaussian() }; 
        }
        for(;i < N; i++){
            inputData[i] = new Double[]{null};
        }
    }

    private void initializeOutputData() {
        outputData = new Double[N][1];
        int i;
        for(i = 0; i < n_depth; i++){
            outputData[i] = new Double[]{null};
        }
        for(; i < N; i++){
            outputData[i] = new Double[]{gaussNull(inputData[i-n_depth][0])};
        }
    }

    private static Double gaussNull(double x){
        double chance = Math.exp(-x*x/((1)*2));
        return rng.nextDouble() < chance ? x + 1 : null;
    }

    public static void main(String[] args) {
        MultiLayerPeakTest ptest = new MultiLayerPeakTest();

        GraphNetwork net = new GraphNetwork();

        NodeBuilder nodeBuilder = new NodeBuilder(net);

        nodeBuilder.setActivationFunction(ActivationFunction.LINEAR);
        nodeBuilder.setNodeConstructor(ComplexNode::new);
        nodeBuilder.setOutputDistSupplier(NormalDistribution::getStandardNormalDistribution);
        // nodeBuilder.setOutputDistAdjusterSupplier(NormalDistributionFromData::new);
        nodeBuilder.setProbabilityDistSupplier(BetaDistribution::getUniformBetaDistribution);
        nodeBuilder.setProbabilityDistAdjusterSupplier(BetaDistributionAdjuster::new);
        // nodeBuilder.setProbabilityDistSupplier(BernoulliDistribution::getEvenDistribution);
        // nodeBuilder.setProbabilityDistAdjusterSupplier(BernoulliDistributionAdjuster::new);

        nodeBuilder.setAsInputNode();

        InputNode in = (InputNode) nodeBuilder.build();

        nodeBuilder.setAsHiddenNode();

        ITrainable hidden1 = (ITrainable) nodeBuilder.build();
        ITrainable hidden2 = (ITrainable) nodeBuilder.build();

        nodeBuilder.setAsOutputNode();

        OutputNode out = (OutputNode) nodeBuilder.build();

        in.setName("Input");
        out.setName("Output");

        ArcBuilder arcBuilder = new ArcBuilder(net);
        arcBuilder.setFilterSupplier(NormalPeakFilter::getStandardNormalBetaFilter);
        arcBuilder.setFilterAdjusterSupplier(NormalBetaFilterAdjuster::new);
        // arcBuilder.setFilterAdjusterSupplier(NormalBernoulliFilterAdjuster::new);
        // arcBuilder.setFilterSupplier(OpenFilter::new);
        // arcBuilder.setFilterAdjusterSupplier(NoAdjustments::new);

        arcBuilder.build(in, hidden1);
        arcBuilder.build(hidden1, hidden2);
        arcBuilder.build(hidden2, out);

        // ADAMTrainer adam = new ADAMTrainer(net, new
        // ErrorFunction.MeanSquaredError());
        // adam.alpha = 0.1;
        // adam.epsilon = 0.01;
        // adam.beta_1 = 0.9;
        // adam.beta_2 = 0.99;

        BackpropTrainer adam = new BackpropTrainer(net, new ErrorFunction.MeanSquaredError(), true);

        adam.setTrainingData(ptest.inputData, ptest.outputData);
        adam.epsilon = 1;

        adam.trainNetwork(20000, 10);
        net.deactivateAll();
        net.setInputOperation(nodeMap -> BackpropTrainer.applyInputToNode(nodeMap, ptest.inputData, counter++));
        for (int i = 0; i < ptest.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
