package com.lucasbrown.InferenceAndAssessment;

import java.util.ArrayList;
import java.util.Objects;

import com.lucasbrown.GraphNetwork.Global.DataGraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Local.IInputNode;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.GraphNetwork.Local.DataStructure.InputDataNode;
import com.lucasbrown.GraphNetwork.Local.DataStructure.OutputDataNode;
import com.lucasbrown.NetworkTraining.GeneticAlgorithm.GeneticTrainer;
import com.lucasbrown.NetworkTraining.GeneticAlgorithm.IGeneticTrainable;

public class GeneticAlgorithmTest {
    
    private static int input_id;
    private static int output_id;

    public static void main(String[] args)
    {
        // basic network setup
        DataGraphNetwork net = new DataGraphNetwork();

        InputDataNode in = net.createInputNode(ActivationFunction.LINEAR);
        OutputDataNode out = net.createOutputNode(ActivationFunction.LINEAR);
        Node hidden = net.createHiddenNode(ActivationFunction.LINEAR);

        in.setName("Input");
        out.setName("Output");
        hidden.setName("Hidden");

        net.addNewConnection(in, hidden, new BellCurveDistribution(0, 1));
        net.addNewConnection(hidden, out, new BellCurveDistribution(0, 1));

        input_id = in.id;
        output_id = out.id;

        GeneticTrainer gt = new GeneticTrainer(10, 25, GeneticAlgorithmTest::fitnessOfNetwork);
        gt.populateRandom(net, ActivationFunction.LINEAR);

        for (int i = 0; i < 10000; i++) {
            gt.getNextPopulation();
        }

        DataGraphNetwork best = (DataGraphNetwork) gt.population[0];

        // just test inference
        best.setInputOperation(input_nodes -> in.acceptUserForwardSignal(1));
        best.setOutputOperation(output_nodes -> System.out.println(net.toString()));

        for(int i = 0; i < 25; i++){
            best.inferenceStep();
        }
    }

    private static double fitnessOfNetwork(IGeneticTrainable individual)
    {
        DataGraphNetwork net = (DataGraphNetwork) individual;
        int collection_points = 10;
        double target = -4;
        ArrayList<Double> outputs = new ArrayList<>(collection_points);

        net.setInputOperation(input_nodes -> input_nodes.get(input_id).acceptUserForwardSignal(1));
        net.setOutputOperation(output_nodes -> outputs.add(output_nodes.get(output_id).getValueOrNull()));

        // test the network
        for (int i = 0; i < collection_points; i++) {
            net.inferenceStep();
        }

        long null_counts = outputs.stream().filter(Objects::isNull).count();
        if(null_counts == collection_points)
        {
            return Double.MAX_VALUE;
        }

        double score = outputs.stream()
                .filter(Objects::nonNull)
                .mapToDouble(d -> (d - target) * (d-target))
                .sum();

        score *= 1 + null_counts - 2; // 2 nulls are to be expected
        return score;
    }

}
