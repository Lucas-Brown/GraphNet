package com.lucasbrown.NetworkTraining;

import java.util.Random;
import java.util.function.Supplier;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.NodeBuilder;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Filters.CappedNormalPeakFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.GeneralizedExponentialDecayFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.NormalPeakFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.OpenFilter;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators.ComplexProbabilityCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators.IProbabilityCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators.SimpleProbabilityCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.ComplexCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.IValueCombinator;
import com.lucasbrown.NetworkTraining.Solvers.ADAMSolver;
import com.lucasbrown.NetworkTraining.Trainers.Trainer;

public class MultiplierTest {

    private Random rng = new Random();
    private int[] M1 = {10,1,2,3,4,5,6,7,8,9};
    private int[] M2 = {70,10,20,30,40,50,60,80,90,100};
    private int N = M1.length * M2.length;
    private final int mul_steps = 3;
    private final int overshoots = 5;
    private Double[][][] inputData;
    private Double[][][] outputData;

    private void initializeInputData() {
        inputData = new Double[N][][];

        int linearIdx = 0;
        for (int i = 0; i < M1.length; i++) {
            for (int j = 0; j < M2.length; j++) {
                inputData[linearIdx++] = createInput(M1[i], M2[j]);
            }
        }
    }

    private Double[][] createInput(int m1, int m2){
        Double[][] input = new Double[m1+mul_steps+overshoots][2];
        input[0] = new Double[] { (double) m1, (double) m2 };
        for (int i = 1; i < input.length; i++) {
            input[i] = new Double[] { null, null };
        }
        return input;
    }


    private void initializeOutputData() {
        outputData = new Double[N][][];
        for(int i = 0; i < N; i++){
            outputData[i] = createOutput(inputData[i]);
        }
    }

    private Double[][] createOutput(Double[][] input){
        int m1 = (int) input[0][0].longValue();
        int m2 = (int) input[0][1].longValue();

        Double[][] out = new Double[m1+mul_steps+overshoots][1];
        for (int i = 0; i < input.length; i++) {
            out[i] = new Double[]{null};
        }
        out[out.length-overshoots-1] = new Double[]{(double) m1*m2};
        return out;
    }

    public GraphNetwork initializeNetwork(){
        GraphNetwork net = new GraphNetwork();

        NodeBuilder nodeBuilder = new NodeBuilder(net);

        nodeBuilder.setActivationFunction(ActivationFunction.LINEAR);
        nodeBuilder.setValueCombinator(ComplexCombinator::new);
        nodeBuilder.setProbabilityCombinator(() -> new SimpleProbabilityCombinator(OpenFilter::new));
        
        nodeBuilder.setAsInputNode();
        InputNode in1 = (InputNode) nodeBuilder.build();
        InputNode in2 = (InputNode) nodeBuilder.build();

        nodeBuilder.setAsHiddenNode();
        INode counterNode = nodeBuilder.build();
        INode registerNode = nodeBuilder.build();
        INode valueAccumulator = nodeBuilder.build();

        nodeBuilder.setAsOutputNode();
        nodeBuilder.setProbabilityCombinator(() -> new ComplexProbabilityCombinator(NormalPeakFilter::getStandardNormalBetaFilter));
        OutputNode out = (OutputNode) nodeBuilder.build();

        in1.setName("Input 1");
        in2.setName("Input 2");
        counterNode.setName("Counter");
        registerNode.setName("Register");
        valueAccumulator.setName("Value");
        out.setName("Output");

        net.addNewConnection(in1, counterNode);
        net.addNewConnection(in2, registerNode);
        net.addNewConnection(counterNode, counterNode);
        net.addNewConnection(registerNode, registerNode);
        net.addNewConnection(registerNode, valueAccumulator);
        net.addNewConnection(valueAccumulator, valueAccumulator);
        net.addNewConnection(counterNode, out);
        net.addNewConnection(valueAccumulator, out);


        // rigging the odds
        IValueCombinator counterVComb = counterNode.getValueCombinator();
        counterVComb.setBias(0b01, 0);
        counterVComb.setWeights(0b01, new double[]{1});
        counterVComb.setBias(0b10, -1);
        counterVComb.setWeights(0b10, new double[]{1});

        
        IValueCombinator registerVComb = registerNode.getValueCombinator();
        registerVComb.setBias(0b01, 0);
        registerVComb.setWeights(0b01, new double[]{1});
        registerVComb.setBias(0b10, 0);
        registerVComb.setWeights(0b10, new double[]{1});


        IValueCombinator valueVComb = valueAccumulator.getValueCombinator();
        valueVComb.setBias(0b01, 0);
        valueVComb.setWeights(0b01, new double[]{1});
        valueVComb.setBias(0b11, 0);
        valueVComb.setWeights(0b11, new double[]{1,1});

        
        IValueCombinator outVComb = out.getValueCombinator();
        outVComb.setBias(0b11, 0);
        outVComb.setWeights(0b11, new double[]{0, 1});

        IProbabilityCombinator outPComb = out.getProbabilityCombinator();
        IFilter[] outFilters = outPComb.getFilters(0b11);
        // outFilters[0].setAdjustableParameters(new double[]{-10, 10, 0, 1, 2});
        // outFilters[1].setAdjustableParameters(new double[]{-10, 10, 0, 1000, 2});
        outFilters[1].setAdjustableParameters(100,1000);
        outFilters[0].setAdjustableParameters(0,1);

        return net;
    }


    public static void main(String[] args) {
        MultiplierTest mul = new MultiplierTest();
        mul.initializeInputData();
        mul.initializeOutputData();

        GraphNetwork net = mul.initializeNetwork();
        Trainer trainer = Trainer.getDefaultTrainer(net);
        trainer.setTrainingData(mul.inputData, mul.outputData);

        ADAMSolver weightsSolver = (ADAMSolver) trainer.weightsSolver;
        weightsSolver.alpha= 1E-12;
        weightsSolver.epsilon = 1E-24;

        ADAMSolver probsSolver = (ADAMSolver) trainer.probabilitySolver;
        probsSolver.alpha = 1E1;
        probsSolver.epsilon = 1E-6;
        // probsSolver.beta_1=0.999;
        // probsSolver.beta_2=0.9999;

        trainer.trainNetwork(10000000, 100);

        net.deactivateAll();
       
    }
}
