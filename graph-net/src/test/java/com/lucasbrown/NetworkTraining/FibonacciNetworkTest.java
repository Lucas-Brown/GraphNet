package com.lucasbrown.NetworkTraining;

import com.lucasbrown.GraphNetwork.Distributions.OpenFilter;
import com.lucasbrown.GraphNetwork.Global.BackpropTrainer;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.SimpleNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;

public class FibonacciNetworkTest {

    private static int counter = 0;
    private int N = 15;
    private Double[][] inputData;
    private Double[][] outputData;

    private void initializeInputData() {
        inputData = new Double[N][1];
        inputData[0] = new Double[]{0d};
        for(int i = 1; i < N; i++){
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

        InputNode in = SimpleNode.asInputNode(ActivationFunction.LINEAR);
        SimpleNode hidden = new SimpleNode(ActivationFunction.LINEAR); 
        OutputNode out = SimpleNode.asOutputNode(ActivationFunction.LINEAR);

        in.setName("Input");
        hidden.setName("Hidden");
        out.setName("Output");

        net.addNodeToNetwork(in);
        net.addNodeToNetwork(hidden);
        net.addNodeToNetwork(out);

        net.addNewConnection(in, hidden, new OpenFilter());
        net.addNewConnection(hidden, hidden, new OpenFilter());
        net.addNewConnection(hidden, out, new OpenFilter());

        BackpropTrainer bt = new BackpropTrainer(net, new ErrorFunction.MeanSquaredError());
        bt.epsilon = 0.0001;

        bt.setTrainingData(fibNet.inputData, fibNet.outputData);
        bt.trainNetwork(500000, 10000);

        net.deactivateAll();
        net.setInputOperation(nodeMap -> BackpropTrainer.applyInputToNode(nodeMap, fibNet.inputData, counter++));
        for (int i = 0; i < fibNet.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
