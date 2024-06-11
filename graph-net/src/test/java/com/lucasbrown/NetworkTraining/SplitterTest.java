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
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistributionAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NoAdjustments;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalPeakFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster2;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.OpenFilter;

public class SplitterTest {
    
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
        outputData = new Double[N][2];
        for (int i = 0; i < N; i++) {
            outputData[(i + 1) % N] = new Double[]{Double.valueOf(i), Double.valueOf(2*i - 1)};
        }
    } 

    public static void main(String[] args){
        SplitterTest split = new SplitterTest();
        split.initializeInputData();
        split.initializeOutputData();

        GraphNetwork net = new GraphNetwork();

        NodeBuilder nodeBuilder = new NodeBuilder(net);

        nodeBuilder.setActivationFunction(ActivationFunction.LINEAR);
        nodeBuilder.setNodeConstructor(SimpleNode::new);
        nodeBuilder.setOutputDistSupplier(NormalDistribution::getStandardNormalDistribution);
        nodeBuilder.setProbabilityDistSupplier(BetaDistribution::getUniformBetaDistribution);
        nodeBuilder.setProbabilityDistAdjusterSupplier(BetaDistributionAdjuster::new);

        nodeBuilder.setAsInputNode();
        InputNode in = (InputNode) nodeBuilder.build();

        nodeBuilder.setAsOutputNode();
        OutputNode out1 = (OutputNode) nodeBuilder.build();
        OutputNode out2 = (OutputNode) nodeBuilder.build();

        in.setName("Input");
        out1.setName("Output 1");
        out2.setName("Output 2");

        ArcBuilder arcBuilder = new ArcBuilder(net);
        // arcBuilder.setFilterSupplier(OpenFilter::new);
        // arcBuilder.setFilterAdjusterSupplier(NoAdjustments::new);
        arcBuilder.setFilterSupplier(NormalPeakFilter::getStandardNormalBetaFilter);
        arcBuilder.setFilterAdjusterSupplier(NormalBetaFilterAdjuster2::new);

        arcBuilder.build(in, out1);
        arcBuilder.build(in, out2);

        BackpropTrainer bt = new BackpropTrainer(net, new ErrorFunction.MeanSquaredError());
        bt.epsilon = 0.1;

        bt.setTrainingData(split.inputData, split.outputData);

        bt.trainNetwork(10000, 100);
        net.deactivateAll();
        net.setInputOperation(nodeMap -> BackpropTrainer.applyInputToNode(nodeMap, split.inputData, counter++));
        for (int i = 0; i < split.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
