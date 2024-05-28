package com.lucasbrown.NetworkTraining;

import com.lucasbrown.GraphNetwork.Global.ArcBuilder;
import com.lucasbrown.GraphNetwork.Global.BackpropTrainer;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.NodeBuilder;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.SimpleNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistribution;

public class FibonacciNetworkTest {

    private static int counter = 0;
    private int N = 15;
    private Double[][] inputData;
    private Double[][] outputData;

    private void initializeInputData() {
        inputData = new Double[N][1];
        inputData[0] = new Double[]{0d};
        inputData[1] = new Double[]{1d};
        for(int i = 2; i < N; i++){
            inputData[i] = new Double[]{null}; 
        }
    }

    private void initializeOutputData() {
        double[] sequence = fib();
        outputData = new Double[N][1];
        outputData[0] = new Double[]{null};
        outputData[1] = new Double[]{null};

        for(int i = 2; i < N; i++){
            outputData[i] = new Double[]{sequence[i-2]};
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
        nodeBuilder.setNodeClass(SimpleNode.class);
        nodeBuilder.setOutputDistSupplier(NormalDistribution::getStandardNormalDistribution);
        nodeBuilder.setProbabilityDistSupplier(BetaDistribution::getUniformBetaDistribution);

        nodeBuilder.setAsInputNode();
        InputNode in = (InputNode) nodeBuilder.build();

        nodeBuilder.setAsHiddenNode();
        INode hidden1 = nodeBuilder.build();
        INode hidden2 = nodeBuilder.build();

        nodeBuilder.setAsOutputNode();
        OutputNode out = (OutputNode) nodeBuilder.build();

        in.setName("Input");
        hidden1.setName("Hidden 1");
        hidden2.setName("Hidden 2");
        out.setName("Output");

        ArcBuilder arcBuilder = new ArcBuilder(net);
        arcBuilder.setFilterSupplier(NormalBetaFilter::getStandardNormalBetaFilter);
        arcBuilder.setFilterAdjusterSupplier(NormalBetaFilterAdjuster::new);

        arcBuilder.build(in, hidden1);
        arcBuilder.build(hidden1, hidden2);
        arcBuilder.build(hidden2, hidden1);
        arcBuilder.build(hidden1, out);


        BackpropTrainer bt = new BackpropTrainer(net, new ErrorFunction.MeanSquaredError());
        bt.epsilon = 0.001;

        bt.setTrainingData(fibNet.inputData, fibNet.outputData);
        bt.trainNetwork(10000, 1);

        net.deactivateAll();
        net.setInputOperation(nodeMap -> BackpropTrainer.applyInputToNode(nodeMap, fibNet.inputData, counter++));
        for (int i = 0; i < fibNet.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
