package com.lucasbrown.NetworkTraining;

import java.util.Random;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.NodeBuilder;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Filters.NormalPeakFilter;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators.ComplexProbabilityCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators.SimpleProbabilityCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.ComplexCombinator;
import com.lucasbrown.NetworkTraining.Solvers.ADAMSolver;
import com.lucasbrown.NetworkTraining.Trainers.Trainer;

public class MultiLayerSelectorTest {

    private static Random rng = new Random();
    private static int counter = 0;
    private static int n_depth = 2;
    private static int N = 100;

    private Double[][] inputData;
    private Double[][] outputData;

    private MultiLayerSelectorTest(){
        initializeInputData();
        initializeOutputData();
    }


    private void initializeInputData() {
        inputData = new Double[N][1];
        int i;
        for(i = 0; i < N-n_depth; i++){
            inputData[i] = new Double[]{ rng.nextGaussian()}; 
        }
        for(;i < N; i++){
            inputData[i] = new Double[]{null};
        }
    }

    private void initializeOutputData() {
        outputData = new Double[N][3];
        int i;
        for(i = 0; i < n_depth; i++){
            outputData[i] = new Double[]{null, null, null};
        }
        for(; i < N; i++){
            double x = inputData[i-n_depth][0];
            int target = (int) Math.round(x); 
            outputData[i] = new Double[]{null,null,null};
            if(target > -2 && target < 2){
                outputData[i][target + 1] = 1d;
            }
        }
    }


    public static void main(String[] args) {
        MultiLayerSelectorTest ptest = new MultiLayerSelectorTest();

        GraphNetwork net = new GraphNetwork();
        
        NodeBuilder nodeBuilder = new NodeBuilder(net);

        nodeBuilder.setActivationFunction(ActivationFunction.LINEAR);
        nodeBuilder.setValueCombinator(ComplexCombinator::new);
        nodeBuilder.setProbabilityCombinator(() -> new ComplexProbabilityCombinator(NormalPeakFilter::getStandardNormalBetaFilter));
        
        nodeBuilder.setAsInputNode();

        InputNode in = (InputNode) nodeBuilder.build();

        nodeBuilder.setAsHiddenNode();

        // INode hidden11 = nodeBuilder.build();
        // INode hidden12 = nodeBuilder.build();

        nodeBuilder.setAsOutputNode();

        OutputNode out1 = (OutputNode) nodeBuilder.build();
        OutputNode out2 = (OutputNode) nodeBuilder.build();
        OutputNode out3 = (OutputNode) nodeBuilder.build();

        in.setName("Input");
        out1.setName("Output");
        out2.setName("Output");
        out3.setName("Output");

        // net.addNewConnection(in, hidden11);
        // net.addNewConnection(in, hidden12);

        // net.addNewConnection(hidden11, out1);
        // net.addNewConnection(hidden11, out2);
        // net.addNewConnection(hidden11, out3);
        // net.addNewConnection(hidden12, out1);
        // net.addNewConnection(hidden12, out2);
        // net.addNewConnection(hidden12, out3);

        // net.addNewConnection(hidden11, out1);
        // net.addNewConnection(hidden11, out2);
        // net.addNewConnection(hidden11, out3);
        // net.addNewConnection(hidden12, out1);
        // net.addNewConnection(hidden12, out2);
        // net.addNewConnection(hidden12, out3);

        net.addNewConnection(in, out1);
        net.addNewConnection(in, out2);
        net.addNewConnection(in, out3);

        Trainer trainer = Trainer.getDefaultTrainer(net);
        trainer.setTrainingData(ptest.inputData, ptest.outputData);
        // ADAMSolver weightSolver = (ADAMSolver) trainer.weightsSolver;
        // weightSolver.alpha = 0.1;
        // weightSolver.epsilon = 0.01;
        // weightSolver.beta_1 = 0.9;
        // weightSolver.beta_2 = 0.99;

        // ADAMSolver probabilitySolver = (ADAMSolver) trainer.probabilitySolver;
        // probabilitySolver.alpha = 1E-2;
        // probabilitySolver.epsilon = 1E-10;
        // probabilitySolver.beta_1 = 0.99;
        // probabilitySolver.beta_2 = 0.999;

        trainer.trainNetwork(1000000, 1000);
        System.out.println();
    }
}
