package com.lucasbrown.NetworkTraining;

import com.lucasbrown.GraphNetwork.Global.ArcBuilder;
import com.lucasbrown.GraphNetwork.Global.ArcBuilder.FilterAdjusterFunction;
import com.lucasbrown.GraphNetwork.Global.BackpropTrainer;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.NodeBuilder;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Nodes.ComplexNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.SimpleNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistributionFromData;
import com.lucasbrown.NetworkTraining.DataSetTraining.IExpectationAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.IFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.ITrainableDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistributionFromData;
import com.lucasbrown.NetworkTraining.DataSetTraining.OpenFilter;

public class RepeaterTest {

    private static int counter = 0;
    private int N = 15;
    private Double[][] inputData;
    private Double[][] outputData;

    private void initializeInputData() {
        inputData = new Double[N][1];
        inputData[0] = new Double[]{2d};
        for(int i = 1; i < N; i++){
            inputData[i] = new Double[]{null}; 
        }
    }

    private void initializeOutputData() {
        outputData = new Double[N][1];
        outputData[0] = new Double[]{null};
        outputData[1] = new Double[]{null};

        for(int i = 2; i < N; i++){
            outputData[i] = new Double[]{(double) i};
        }
    }

    public static void main(String[] args) {
        RepeaterTest repeater = new RepeaterTest();
        repeater.initializeInputData();
        repeater.initializeOutputData();

        GraphNetwork net = new GraphNetwork();

        NodeBuilder nodeBuilder = new NodeBuilder(net);

        nodeBuilder.setActivationFunction(ActivationFunction.LINEAR);
        nodeBuilder.setNodeConstructor(ComplexNode::new);
        nodeBuilder.setOutputDistSupplier(NormalDistribution::getStandardNormalDistribution);
        nodeBuilder.setOutputDistAdjusterSupplier(NormalDistributionFromData::new);
        nodeBuilder.setProbabilityDistSupplier(BetaDistribution::getUniformBetaDistribution);
        nodeBuilder.setProbabilityDistAdjusterSupplier(BetaDistributionFromData::new);

        nodeBuilder.setAsInputNode();
        InputNode in = (InputNode) nodeBuilder.build();

        nodeBuilder.setAsHiddenNode();
        INode hidden = nodeBuilder.build();

        nodeBuilder.setAsOutputNode();
        OutputNode out = (OutputNode) nodeBuilder.build();

        in.setName("Input");
        hidden.setName("Hidden");
        out.setName("Output");

        ArcBuilder arcBuilder = new ArcBuilder(net);
        arcBuilder.setFilterSupplier(NormalBetaFilter::getStandardNormalBetaFilter);
        arcBuilder.setFilterAdjusterSupplier(NormalBetaFilterAdjuster::new);
        // arcBuilder.setFilterSupplier(OpenFilter::new);
        // arcBuilder.setFilterAdjusterSupplier((IFilter filter, ITrainableDistribution dist1, ITrainableDistribution dist2) -> (IExpectationAdjuster) null);

        arcBuilder.build(in, hidden);
        arcBuilder.build(hidden, hidden);
        arcBuilder.build(hidden, out);


        BackpropTrainer bt = new BackpropTrainer(net, new ErrorFunction.MeanSquaredError());
        bt.epsilon = 0.001;

        bt.setTrainingData(repeater.inputData, repeater.outputData);
        bt.trainNetwork(10000, 1);

        net.deactivateAll();
        net.setInputOperation(nodeMap -> BackpropTrainer.applyInputToNode(nodeMap, repeater.inputData, counter++));
        for (int i = 0; i < repeater.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
