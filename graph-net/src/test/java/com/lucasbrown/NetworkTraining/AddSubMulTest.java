package com.lucasbrown.NetworkTraining;

import java.util.Objects;
import java.util.Random;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Global.EdgeBuilder;
import com.lucasbrown.GraphNetwork.Global.BackpropTrainer;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.NodeBuilder;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Edge;
import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.NormalPeakFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.OpenFilter;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.SimpleNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.ComplexCombinator;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BernoulliDistribution;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BernoulliDistributionAdjuster;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BetaDistribution;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BetaDistributionAdjuster2;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BetaDistributionFromData;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.BetaDistributionFromData2;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.IExpectationAdjuster;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.ITrainableDistribution;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NoAdjustments;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NormalBernoulliFilterAdjuster;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NormalBetaFilterAdjuster;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NormalDistribution;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.NormalDistributionFromData;
import com.lucasbrown.NetworkTraining.OutputDerivatives.ErrorFunction;

public class AddSubMulTest {

    private static Random rng = new Random();
    private static int counter = 0;

    private int N = 100;
    private Double[][] inputData;
    private Double[][] outputData;

    private void initializeInputData() {
        inputData = new Double[N][1];
        for (int i = 0; i < N; i++) {
            inputData[i] = new Double[] { rng.nextGaussian(), rng.nextGaussian()};
            // if(inputData[i][0] == null & inputData[i][1] == null & inputData[i][2] ==
            // null){
            // i--;
            // }
        }
    }

    private void initializeOutputData() {
        outputData = new Double[N][3];
        for (int i = 0; i < N; i++) {
            Double[] data = inputData[i];
            outputData[(i + 3) % N] = new Double[]{
                data[0] + data[1],
                data[0] - data[1],
                data[1] - data[0]
            };
        }
    }

    public static void main(String[] args) {
        AddSubMulTest operator = new AddSubMulTest();
        operator.initializeInputData();
        operator.initializeOutputData();

        GraphNetwork net = new GraphNetwork();

        NodeBuilder nodeBuilder = new NodeBuilder(net);

        nodeBuilder.setActivationFunction(ActivationFunction.LINEAR);
        nodeBuilder.setNodeConstructor(SimpleNode::new);
        nodeBuilder.setOutputDistSupplier(NormalDistribution::getStandardNormalDistribution);
        // nodeBuilder.setOutputDistAdjusterSupplier(NormalDistributionFromData::new);
        nodeBuilder.setProbabilityDistSupplier(BernoulliDistribution::getEvenDistribution);
        nodeBuilder.setProbabilityDistAdjusterSupplier(BernoulliDistributionAdjuster::new);

        nodeBuilder.setAsInputNode();

        InputNode in1 = (InputNode) nodeBuilder.build();
        InputNode in2 = (InputNode) nodeBuilder.build();

        nodeBuilder.setAsHiddenNode();

        INode hidden1_1 = nodeBuilder.build();
        INode hidden1_2 = nodeBuilder.build();
        INode hidden1_3 = nodeBuilder.build();
        INode hidden2_1 = nodeBuilder.build();
        INode hidden2_2 = nodeBuilder.build();
        INode hidden2_3 = nodeBuilder.build();

        nodeBuilder.setAsOutputNode();

        OutputNode out1 = (OutputNode) nodeBuilder.build();
        OutputNode out2 = (OutputNode) nodeBuilder.build();
        OutputNode out3 = (OutputNode) nodeBuilder.build();

        EdgeBuilder arcBuilder = new EdgeBuilder(net);
        // arcBuilder.setFilterSupplier(OpenFilter::new);
        // arcBuilder.setFilterAdjusterSupplier(NoAdjustments::new);
        arcBuilder.setFilterSupplier(NormalPeakFilter::getStandardNormalBetaFilter);
        arcBuilder.setFilterAdjusterSupplier(NormalBernoulliFilterAdjuster::new);

        // may need to write a dense layer builder
        arcBuilder.build(in1, hidden1_1);
        arcBuilder.build(in1, hidden1_2);
        arcBuilder.build(in1, hidden1_3);
        arcBuilder.build(in2, hidden1_1);
        arcBuilder.build(in2, hidden1_2);
        arcBuilder.build(in2, hidden1_3);

        arcBuilder.build(hidden1_1, hidden2_1);
        arcBuilder.build(hidden1_1, hidden2_2);
        arcBuilder.build(hidden1_1, hidden2_3);
        arcBuilder.build(hidden1_2, hidden2_1);
        arcBuilder.build(hidden1_2, hidden2_2);
        arcBuilder.build(hidden1_2, hidden2_3);
        arcBuilder.build(hidden1_3, hidden2_1);
        arcBuilder.build(hidden1_3, hidden2_2);
        arcBuilder.build(hidden1_3, hidden2_3);
        

        arcBuilder.build(hidden2_1, out1);
        arcBuilder.build(hidden2_2, out1);
        arcBuilder.build(hidden2_3, out1);
        arcBuilder.build(hidden2_1, out2);
        arcBuilder.build(hidden2_2, out2);
        arcBuilder.build(hidden2_3, out2);
        arcBuilder.build(hidden2_1, out3);
        arcBuilder.build(hidden2_2, out3);
        arcBuilder.build(hidden2_3, out3);

        BackpropTrainer bt = new BackpropTrainer(net, new ErrorFunction.MeanSquaredError());
        bt.epsilon = 0.1;

        bt.setTrainingData(operator.inputData, operator.outputData);

        bt.trainNetwork(10000, 5);
        net.deactivateAll();
        net.setInputOperation(nodeMap -> BackpropTrainer.applyInputToNode(nodeMap, operator.inputData, counter++));
        for (int i = 0; i < operator.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
