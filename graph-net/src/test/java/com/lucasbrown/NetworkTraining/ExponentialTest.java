package com.lucasbrown.NetworkTraining;

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
import com.lucasbrown.NetworkTraining.DataSetTraining.BernoulliDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.BernoulliDistributionAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.FlatRateFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.NoAdjustments;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBernoulliFilterAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalPeakFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistribution;

public class ExponentialTest {

    private static int counter = 0;
    private static double base = 2;
    private int N = 25;
    private Double[][] inputData;
    private Double[][] outputData;

    private void initializeInputData() {
        inputData = new Double[N][1];
        inputData[0] = new Double[] { 2d };
        for (int i = 1; i < N; i++) {
            inputData[i] = new Double[] { null };
        }
    }

    private void initializeOutputData() {
        outputData = new Double[N][1];
        outputData[0] = new Double[] { null };
        outputData[1] = new Double[] { null };
        outputData[2] = new Double[] { 1d };

        for (int i = 3; i < N; i++) {
            outputData[i] = new Double[] { outputData[i - 1][0] * base };
        }
    }

    public static void main(String[] args) {
        ExponentialTest exponentialGrowth = new ExponentialTest();
        exponentialGrowth.initializeInputData();
        exponentialGrowth.initializeOutputData();

        GraphNetwork net = new GraphNetwork();

        NodeBuilder nodeBuilder = new NodeBuilder(net);

        nodeBuilder.setActivationFunction(ActivationFunction.LINEAR);
        nodeBuilder.setNodeConstructor(ComplexNode::new);
        nodeBuilder.setOutputDistSupplier(NormalDistribution::getStandardNormalDistribution);
        // nodeBuilder.setOutputDistAdjusterSupplier(NormalDistributionFromData::new);
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
        arcBuilder.setFilterSupplier(NormalPeakFilter::getStandardNormalBetaFilter);
        arcBuilder.setFilterAdjusterSupplier(NormalBernoulliFilterAdjuster::new);
        // arcBuilder.setFilterSupplier(() -> new FlatRateFilter(0.999));
        // arcBuilder.setFilterAdjusterSupplier(NoAdjustments::new);

        arcBuilder.build(in, hidden);
        arcBuilder.build(hidden, hidden);
        arcBuilder.build(hidden, out);

        
        Trainer trainer = Trainer.getDefaultTrainer(net, exponentialGrowth.inputData, exponentialGrowth.outputData);
        // ADAMSolver weightSolver = (ADAMSolver) trainer.weightsSolver;
        // weightSolver.alpha = 0.01;
        // weightSolver.epsilon = 0.0001;
        // weightSolver.beta_1 = 0.9;
        // weightSolver.beta_2 = 0.99;

        // ADAMSolver probabilitySolver = (ADAMSolver) trainer.probabilitySolver;
        // probabilitySolver.alpha = 0.01;
        // probabilitySolver.epsilon = 0.0001;
        // probabilitySolver.beta_1 = 0.9;
        // probabilitySolver.beta_2 = 0.99;

        trainer.trainNetwork(100000, 1000);

        // weightSolver.alpha = 0.01;
        // weightSolver.epsilon = 0.0000001;
        // probabilitySolver.alpha = 0.01;
        // probabilitySolver.epsilon = 0.0000001;

        // trainer.trainNetwork(100000, 1000);

        net.deactivateAll();
       
    }
}
