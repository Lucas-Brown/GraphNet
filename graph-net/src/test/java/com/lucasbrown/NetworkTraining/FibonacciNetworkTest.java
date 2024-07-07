package com.lucasbrown.NetworkTraining;

import com.lucasbrown.GraphNetwork.Global.ArcBuilder;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.NodeBuilder;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Filters.FlatRateFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.NormalPeakFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.OpenFilter;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.SimpleNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.ComplexCombinator;
import com.lucasbrown.NetworkTraining.DataSetTraining.AmplifierFilter;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BernoulliDistribution;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BernoulliDistributionAdjuster;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NoAdjustments;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NormalBernoulliFilterAdjuster;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NormalDistribution;
import com.lucasbrown.NetworkTraining.Solvers.ADAMSolver;
import com.lucasbrown.NetworkTraining.Trainers.Trainer;

public class FibonacciNetworkTest {

    private int offset = 2;
    private int N = 10;
    private Double[][][] inputData;
    private Double[][][] outputData;

    
    private void initializeInputData() {
        inputData = new Double[][][]{ createInput(0, 1), createInput(1, -1), createInput(1, 1) };
    }

    private Double[][] createInput(double init_1, double init_2) {
        Double[][] in = new Double[N + offset][1];
        in[0] = new Double[] { init_1 };
        in[1] = new Double[] { init_2 };
        for (int i = offset; i < N + offset; i++) {
            in[i] = new Double[] { null };
        }
        return in;
    }
    
    private void initializeOutputData() {
        outputData = new Double[inputData.length][][];
        for(int i = 0; i < inputData.length; i++){
            outputData[i] = createOutput(inputData[i]);
        }
    }

    private Double[][] createOutput(Double[][] input) {
        double[] sequence = fib(input[0][0], input[1][0]);
        Double[][] out = new Double[N + offset][1];
        int i = 0;
        for (; i < offset; i++) {
            out[i] = new Double[] { null };
        }

        for (; i < N + offset; i++) {
            out[i] = new Double[] { sequence[i - offset] };
        }
        return out;
    }

    private double[] fib(double init_1, double init_2) {
        double[] sequence = new double[N];
        sequence[0] = init_1;
        sequence[1] = init_2;
        for (int i = 2; i < N; i++) {
            sequence[i] = sequence[i - 1] + sequence[i - 2];
        }
        return sequence;
    }

    public static void main(String[] args) {
        FibonacciNetworkTest fibNet = new FibonacciNetworkTest();
        fibNet.initializeInputData();
        fibNet.initializeOutputData();

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
        ITrainable hidden1 = (ITrainable) nodeBuilder.build();
        ITrainable hidden2 = (ITrainable) nodeBuilder.build();

        nodeBuilder.setAsOutputNode();
        OutputNode out = (OutputNode) nodeBuilder.build();

        in.setName("Input");
        hidden1.setName("Hidden 1");
        hidden2.setName("Hidden 2");
        out.setName("Output");

        ArcBuilder arcBuilder = new ArcBuilder(net);
        // arcBuilder.setFilterSupplier(() -> new AmplifierFilter(0.999, 1.5));
        arcBuilder.setFilterSupplier(() -> new FlatRateFilter(0.5));
        // arcBuilder.setFilterSupplier(OpenFilter::new);
        arcBuilder.setFilterAdjusterSupplier(NoAdjustments::new);
        // arcBuilder.setFilterSupplier(NormalPeakFilter::getStandardNormalBetaFilter);
        // arcBuilder.setFilterAdjusterSupplier(NormalBernoulliFilterAdjuster::new);

        arcBuilder.build(in, hidden1);  
        arcBuilder.build(hidden1, hidden2);
        arcBuilder.build(hidden2, hidden1);
        arcBuilder.build(hidden1, hidden1);
        arcBuilder.build(hidden1, out);

        Trainer trainer = Trainer.getDefaultTrainer(net);
        trainer.setTrainingData(fibNet.inputData, fibNet.outputData);
        // ADAMSolver weightSolver = (ADAMSolver) trainer.weightsSolver;
        // weightSolver.alpha = 0.1;

        // ADAMSolver probabilitySolver = (ADAMSolver) trainer.probabilitySolver;
        // probabilitySolver.alpha = 0.1;

        trainer.trainNetwork(10000, 200);
        System.out.println();
    }
}
