package com.lucasbrown.NetworkTraining.OutputDerivatives;

import org.junit.Assert;
import org.junit.Test;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.NodeBuilder;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Filters.NormalPeakFilter;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators.ComplexProbabilityCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.ComplexCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.IValueCombinator;
import com.lucasbrown.NetworkTraining.History.NetworkHistory;
import com.lucasbrown.NetworkTraining.Trainers.NumericalDerivativeTrainer;
import com.lucasbrown.NetworkTraining.Trainers.Trainer;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class WeightedOutcomeChanceFilterGradientTest {

    private double delta = 1E-6;

    private GraphNetwork getSingleLayerModel(){
        GraphNetwork net = new GraphNetwork();
        
        NodeBuilder nodeBuilder = new NodeBuilder(net);

        nodeBuilder.setActivationFunction(ActivationFunction.LINEAR);
        nodeBuilder.setValueCombinator(ComplexCombinator::new);
        nodeBuilder.setProbabilityCombinator(() -> new ComplexProbabilityCombinator(NormalPeakFilter::getStandardNormalBetaFilter));
        
        nodeBuilder.setAsInputNode();

        InputNode in = (InputNode) nodeBuilder.build();

        nodeBuilder.setAsOutputNode();

        OutputNode out1 = (OutputNode) nodeBuilder.build();
        OutputNode out2 = (OutputNode) nodeBuilder.build();
        OutputNode out3 = (OutputNode) nodeBuilder.build();

        in.setName("Input");
        out1.setName("Output 3");
        out2.setName("Output 2");
        out3.setName("Output 1");

        net.addNewConnection(in, out1);
        net.addNewConnection(in, out2);
        net.addNewConnection(in, out3);

        IValueCombinator vComb1 = out1.getValueCombinator();
        vComb1.setBias(0b1, 0);
        vComb1.setWeights(0b1, new double[]{1});
        
        IValueCombinator vComb2 = out2.getValueCombinator();
        vComb2.setBias(0b1, 0);
        vComb2.setWeights(0b1, new double[]{1});
        
        IValueCombinator vComb3 = out3.getValueCombinator();
        vComb3.setBias(0b1, 0);
        vComb3.setWeights(0b1, new double[]{1});

        return net;
    }

    @Test
    public void testComputeGradient() {
        GraphNetwork net = getSingleLayerModel();

        final Double[][] inputData = {{0.5d}, {null}};
        final Double[][] outputData = {{null, null, null}, {1d, null, null}};
        final double[] targetErrors = {-0.9411918, -0.4705959, 3.31076541, 1.6553827, 3.31076541, 1.6553827};
        
        // numerical check
        NumericalDerivativeTrainer numericalTrainer = NumericalDerivativeTrainer.getDefaultTrainer(net);
        numericalTrainer.setTrainingData(inputData, outputData);

        Vec filterDerivative = numericalTrainer.computeNumericalDerivativeOfFilters();

        // analytic check
        Trainer analyticTrainer = Trainer.getDefaultTrainer(net);
        analyticTrainer.setTrainingData(inputData, outputData);

        NetworkHistory[] histories = analyticTrainer.computeAllHistories();
        
        Vec weightsGradient = analyticTrainer.aggregateWeightGradients(histories);
        Vec probabilityGradient = analyticTrainer.aggregateProbabilityGradients(histories);

        Assert.assertArrayEquals(targetErrors, filterDerivative.arrayCopy(), 1E-6);
        Assert.assertArrayEquals(targetErrors, probabilityGradient.arrayCopy(), 1E-6);

    }
}
