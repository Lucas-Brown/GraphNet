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
import com.lucasbrown.NetworkTraining.History.NetworkHistory;
import com.lucasbrown.NetworkTraining.Trainers.Trainer;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class WeightedOutcomeChanceFilterGradientTest {

    private double delta = 1E-6;

    private static final Double[][] inputData = {{1d}, {null}};
    private static final Double[][] outputData = {{null, null, null}, {null, 1d, null}};

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
        return net;
    }

    @Test
    public void testComputeGradient() {
        GraphNetwork net = getSingleLayerModel();

        

        Trainer trainer = Trainer.getDefaultTrainer(net);
        trainer.setTrainingData(inputData, outputData);

        NetworkHistory[] histories = trainer.computeAllHistories();
        
        Vec weightsGradient = trainer.aggregateWeightGradients(histories);
        Vec probabilityGradient = trainer.aggregateProbabilityGradients(histories);

        double[] linearWeightsAndBias = trainer.weightLinearizer.getAllParameters();
        double[] weightsGradientNumerical = new double[linearWeightsAndBias.length];
        for(int i = 0; i < linearWeightsAndBias.length; i++){
            trainer.weightLinearizer.setParameter(i, linearWeightsAndBias[i] + delta);
            NetworkHistory history1 = trainer.computeAllHistories()[0];
            double error1 = trainer.weightsGradient.getTotalError(history1);

            trainer.weightLinearizer.setParameter(i, linearWeightsAndBias[i] - delta);
            NetworkHistory history2 = trainer.computeAllHistories()[0];
            double error2 = trainer.weightsGradient.getTotalError(history2);

            weightsGradientNumerical[i] = (error1 - error2)/(2*delta);
        }

        Assert.assertArrayEquals(weightsGradient.arrayCopy(), weightsGradientNumerical, 1E-6);
        
    }
}
