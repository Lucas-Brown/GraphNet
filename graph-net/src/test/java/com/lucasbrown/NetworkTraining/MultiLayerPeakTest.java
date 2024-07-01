package com.lucasbrown.NetworkTraining;

import java.util.Random;
import com.lucasbrown.GraphNetwork.Global.Network.ArcBuilder;
import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.Network.NodeBuilder;
import com.lucasbrown.GraphNetwork.Global.Trainers.ADAMSolver;
import com.lucasbrown.GraphNetwork.Global.Trainers.Trainer;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Nodes.ComplexNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistributionAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalPeakFilter;
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
            inputData[i] = new Double[]{ rng.nextGaussian() + 2}; 
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
        double chance = Math.exp(-(x-2)*(x-2)/((1)*2));
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

        Trainer trainer = Trainer.getDefaultTrainer(net, ptest.inputData, ptest.outputData);
        ADAMSolver weightSolver = (ADAMSolver) trainer.weightsSolver;
        weightSolver.alpha = 0.1;
        weightSolver.epsilon = 0.01;
        weightSolver.beta_1 = 0.9;
        weightSolver.beta_2 = 0.99;

        ADAMSolver probabilitySolver = (ADAMSolver) trainer.probabilitySolver;
        probabilitySolver.alpha = 0.1;
        probabilitySolver.epsilon = 0.01;
        probabilitySolver.beta_1 = 0.9;
        probabilitySolver.beta_2 = 0.99;

        trainer.trainNetwork(10000, 1000);
        System.out.println();
    }
}
