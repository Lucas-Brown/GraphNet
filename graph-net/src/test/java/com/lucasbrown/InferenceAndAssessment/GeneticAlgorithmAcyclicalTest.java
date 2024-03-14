package com.lucasbrown.InferenceAndAssessment;

import java.util.ArrayList;
import java.util.Objects;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

import com.lucasbrown.GraphNetwork.Global.DataGraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Local.IInputNode;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.GraphNetwork.Local.DataStructure.InputDataNode;
import com.lucasbrown.GraphNetwork.Local.DataStructure.OutputDataNode;
import com.lucasbrown.NetworkTraining.GeneticAlgorithm.GeneticTrainer;
import com.lucasbrown.NetworkTraining.GeneticAlgorithm.IGeneticTrainable;

public class GeneticAlgorithmAcyclicalTest {

    private static int input_id;
    private static int output_id;

    public static void main(String[] args) {
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

        input_id = in.getID();
        output_id = out.getID();

        GeneticTrainer gt = new GeneticTrainer(100, 1000, GeneticAlgorithmAcyclicalTest::sequenceFitness);
        gt.populateRandom(net, ActivationFunction.LINEAR);
        gt.setStructuralChanges(false);

        for (int i = 0; i < 2000; i++) {
            gt.getNextPopulation();
        }

        DataGraphNetwork best = (DataGraphNetwork) gt.population[0];

        // just test inference
        InputGenerator gen = new InputGenerator();

        best.setInputOperation(input_nodes -> input_nodes.get(input_id).acceptUserForwardSignal(gen.getAsDouble()));
        best.setOutputOperation(output_nodes -> {
        });

        best.clearAllSignals();
        for (int i = 0; i < 25; i++) {
            best.inferenceStep();
            System.out.println(best);
        }
    }

    private static double sequenceFitness(IGeneticTrainable individual) {
        int trials = 5;
        double total_score = 0;
        double null_penalty = 10;
        DataGraphNetwork net = (DataGraphNetwork) individual;
        double[] targets = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        int collection_points = targets.length;

        for (; trials > 0; trials--) {
            net.clearAllSignals();
            ArrayList<Double> outputs = new ArrayList<>(collection_points);

            InputGenerator gen = new InputGenerator();

            net.setInputOperation(input_nodes -> input_nodes.get(input_id).acceptUserForwardSignal(gen.getAsDouble()));
            net.setOutputOperation(output_nodes -> outputs.add(output_nodes.get(output_id).getValueOrNull()));

            // test the network
            for (int i = 0; i < collection_points; i++) {
                net.inferenceStep();
            }

            long null_counts = outputs.stream().filter(Objects::isNull).count();
            if (null_counts == collection_points) {
                return Double.MAX_VALUE;
            }

            double score = IntStream.range(0, collection_points)
                    .mapToDouble(i -> {
                        Double d = outputs.get(i);
                        return Objects.isNull(d) ? 0 : Math.pow(d - targets[i], 2);
                    })
                    .sum();

            score += (null_counts - 2) * null_penalty; // at least 1 null is expected
            total_score += score;
        }
        return total_score;
    }

    private static class InputGenerator implements DoubleSupplier {
        int input_num = 0;

        @Override
        public double getAsDouble() {
            return input_num++;
        }
    }

}
