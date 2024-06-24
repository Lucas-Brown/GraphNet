package com.lucasbrown.NetworkTraining;

import com.lucasbrown.GraphNetwork.Global.Network.ArcBuilder;
import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.Network.NodeBuilder;
import com.lucasbrown.GraphNetwork.Global.Trainers.ADAMTrainer;
import com.lucasbrown.GraphNetwork.Global.Trainers.BackpropTrainer;
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
import com.lucasbrown.NetworkTraining.DataSetTraining.IFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.ITrainableDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBernoulliFilterAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalPeakFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster2;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.OpenFilter;

public class FibonacciNetworkTest {

    private static int counter = 0;
    private int offset = 2;
    private int N = 20;
    private Double[][] inputData;
    private Double[][] outputData;

    private void initializeInputData() {
        inputData = new Double[N + offset][1];
        inputData[0] = new Double[]{0d};
        inputData[1] = new Double[]{1d};
        for(int i = 2; i < N+offset; i++){
            inputData[i] = new Double[]{null}; 
        }
    }

    private void initializeOutputData() {
        double[] sequence = fib();
        outputData = new Double[N+offset][1];
        int i = 0;
        for(; i < offset; i++){
            outputData[i] = new Double[]{null};
        } 

        for(; i < N+offset; i++){
            outputData[i] = new Double[]{sequence[i-offset]};
        }
    }

    private double[] fib(){
        double[] sequence = new double[N];
        sequence[0] = 0;
        sequence[1] = 1;
        for(int i = 2; i < N; i++){
            sequence[i] = sequence[i-1] + sequence[i-2];
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
        nodeBuilder.setNodeConstructor(ComplexNode::new);
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
        // arcBuilder.setFilterSupplier(OpenFilter::new);
        // arcBuilder.setFilterAdjusterSupplier((IFilter filter, ITrainableDistribution dist1, ITrainableDistribution dist2) -> null);
        arcBuilder.setFilterSupplier(NormalPeakFilter::getStandardNormalBetaFilter);
        arcBuilder.setFilterAdjusterSupplier(NormalBernoulliFilterAdjuster::new);

        arcBuilder.build(in, hidden1);
        arcBuilder.build(hidden1, hidden2);
        arcBuilder.build(hidden2, hidden1);
        arcBuilder.build(hidden1, hidden1);
        arcBuilder.build(hidden1, out);


        ADAMTrainer adam = new ADAMTrainer(net, new ErrorFunction.MeanSquaredError());
        adam.alpha = 0.1;
        adam.epsilon = 0.1;

        adam.setTrainingData(fibNet.inputData, fibNet.outputData);
        adam.trainNetwork(10000, 100);

        adam.alpha = 0.0001;
        adam.epsilon = 1E-8;

        adam.trainNetwork(10000, 100);

        net.deactivateAll();
        net.setInputOperation(nodeMap -> BackpropTrainer.applyInputToNode(nodeMap, fibNet.inputData, counter++));
        for (int i = 0; i < fibNet.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
