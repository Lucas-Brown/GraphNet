package com.lucasbrown.NetworkTraining;

import com.lucasbrown.GraphNetwork.Global.Network.ArcBuilder;
import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.Network.NodeBuilder;
import com.lucasbrown.GraphNetwork.Global.Trainers.ADAMTrainer;
import com.lucasbrown.GraphNetwork.Global.Trainers.BackpropTrainer;
import com.lucasbrown.GraphNetwork.Global.Trainers.NewtonTrainer;
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
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistributionAdjuster2;
import com.lucasbrown.NetworkTraining.DataSetTraining.BetaDistributionFromData;
import com.lucasbrown.NetworkTraining.DataSetTraining.FlatRateFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.FlatRateFilterBetaAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.IExpectationAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.IFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.ITrainableDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.NoAdjustments;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBernoulliFilterAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalPeakFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalBetaFilterAdjuster2;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistribution;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistributionFromData;
import com.lucasbrown.NetworkTraining.DataSetTraining.OpenFilter;

public class RepeaterTest {

    private static int counter = 0;
    private int N = 25;
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

        ADAMTrainer adam = new ADAMTrainer(net, new ErrorFunction.MeanSquaredError());
        adam.alpha = 0.1;
        adam.epsilon = 0.0001;
        adam.beta_1 = 0.9;
        adam.beta_2 = 0.99;

        adam.setTrainingData(repeater.inputData, repeater.outputData);
        adam.trainNetwork(100000, 1000);

        net.deactivateAll();
        net.setInputOperation(nodeMap -> BackpropTrainer.applyInputToNode(nodeMap, repeater.inputData, counter++));
        for (int i = 0; i < repeater.inputData.length; i++) {
            net.trainingStep();
            System.out.println(net);
        }
    }
}
