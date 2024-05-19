package com.lucasbrown.NetworkTraining;

import com.lucasbrown.GraphNetwork.Global.BackpropTrainer;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.SimpleNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster2;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.OpenFilter;

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

        SimpleNode s_in = new SimpleNode(net, ActivationFunction.LINEAR, new NormalDistribution(0, 1),
                new BetaDistribution(1, 1, 10));
        SimpleNode hidden = new SimpleNode(net, ActivationFunction.LINEAR, new NormalDistribution(0.5, 1),
                new BetaDistribution(1, 1, 10));

        InputNode in = new InputNode(s_in);

        SimpleNode s_out = new SimpleNode(net, ActivationFunction.LINEAR, new NormalDistribution(0, 1),
                new BetaDistribution(1, 1, 10));
        OutputNode out = new OutputNode(s_out);

        in.setName("Input");
        hidden.setName("Hidden");
        out.setName("Output");

        net.addNodeToNetwork(in);
        net.addNodeToNetwork(hidden);
        net.addNodeToNetwork(out);

        NormalBetaFilter b1 = new NormalBetaFilter(0, 1);
        NormalBetaFilter b2 = new NormalBetaFilter(0, 1);
        Arc a1 = net.addNewConnection(in, hidden, b1, new NormalBetaFilterAdjuster2(b1, (NormalDistribution) in.getOutputDistribution(), (BetaDistribution) in.getSignalChanceDistribution()));
        Arc a2 = net.addNewConnection(hidden, out, b2, new NormalBetaFilterAdjuster2(b2, (NormalDistribution) hidden.getOutputDistribution(), (BetaDistribution) hidden.getSignalChanceDistribution()));
        

        // net.addNewConnection(in, hidden, new OpenFilter(), null);
        // net.addNewConnection(hidden, out, new OpenFilter(), null);

        BackpropTrainer bt = new BackpropTrainer(net, new ErrorFunction.MeanSquaredError());
        bt.epsilon = 0.1;

        bt.setTrainingData(linear.inputData, linear.outputData);

        bt.trainNetwork(10000, 100);
        net.deactivateAll();
        net.setInputOperation(nodeMap -> BackpropTrainer.applyInputToNode(nodeMap, linear.inputData, counter++));
        for (int i = 0; i < linear.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
