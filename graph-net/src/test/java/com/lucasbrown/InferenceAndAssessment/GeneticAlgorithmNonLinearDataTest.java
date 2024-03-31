package com.lucasbrown.InferenceAndAssessment;

import java.util.ArrayList;
import java.util.Objects;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

import com.lucasbrown.GraphNetwork.Distributions.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Global.DataGraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.GraphNetwork.Local.DataStructure.InputDataNode;
import com.lucasbrown.GraphNetwork.Local.DataStructure.OutputDataNode;
import com.lucasbrown.NetworkTraining.GeneticAlgorithm.GeneticTrainer;
import com.lucasbrown.NetworkTraining.GeneticAlgorithm.IGeneticTrainable;

public class GeneticAlgorithmNonLinearDataTest {

    private static int input_id;
    private static int output_id;

    public static void main(String[] args) {
        // basic network setup
        DataGraphNetwork net = new DataGraphNetwork();

        InputDataNode in = net.createInputNode(ActivationFunction.SIGNED_QUADRATIC);
        OutputDataNode out = net.createOutputNode(ActivationFunction.SIGNED_QUADRATIC);
        Node hidden1 = net.createHiddenNode(ActivationFunction.SIGNED_QUADRATIC);
        Node hidden2 = net.createHiddenNode(ActivationFunction.SIGNED_QUADRATIC);

        in.setName("Input");
        out.setName("Output");
        hidden1.setName("Hidden 1");
        hidden2.setName("Hidden 2");

        net.addNewConnection(in, hidden1, new BellCurveDistribution(0, 1));
        net.addNewConnection(hidden1, hidden1, new BellCurveDistribution(0, 1));
        net.addNewConnection(hidden1, hidden2, new BellCurveDistribution(0, 1));
        net.addNewConnection(hidden2, hidden1, new BellCurveDistribution(0, 1));
        net.addNewConnection(hidden1, out, new BellCurveDistribution(0, 1));
        net.addNewConnection(hidden2, out, new BellCurveDistribution(0, 1));

        input_id = in.getID();
        output_id = out.getID();

        GeneticTrainer gt = new GeneticTrainer(100, 1000, GeneticAlgorithmNonLinearDataTest::nonLinearFitness);
        gt.populateRandom(net, ActivationFunction.LINEAR);
        gt.setStructuralChanges(true);

        for (int i = 0; i < 1000; i++) {
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

    private static double nonLinearFitness(IGeneticTrainable individual) {
        int trials = 5;
        double total_score = 0;
        double missed_penalty = 100;
        DataGraphNetwork net = (DataGraphNetwork) individual;
        double[] targets = new double[] { 0, 1, 1, 2, 3, 5, 8, 13, 21};
        int collection_points = targets.length;

        for (; trials > 0; trials--) {
            net.clearAllSignals();
            ArrayList<Double> outputs = new ArrayList<>(collection_points);

            InputGenerator gen = new InputGenerator();

            net.setInputOperation(input_nodes -> input_nodes.get(input_id).acceptUserForwardSignal(gen.getAsDouble()));
            net.setOutputOperation(output_nodes -> {
                Double outVal = output_nodes.get(output_id).getValueOrNull();
                if (outVal == null) {
                    //net.setInputOperation(input_nodes -> {});
                } else {
                    outputs.add(outVal);
                    net.setInputOperation(
                            input_nodes -> input_nodes.get(input_id).acceptUserForwardSignal(gen.getAsDouble()));
                }
            });

            // test the network
            int count = 0;
            while (outputs.size() < collection_points && count++ < 100) {
                net.inferenceStep();
            }

            double score = IntStream.range(0, outputs.size())
                    .mapToDouble(i -> {
                        Double d = outputs.get(i);
                        return Objects.isNull(d) ? 0 : Math.pow(d - targets[i], 2);
                    })
                    .sum();

            score += missed_penalty*(collection_points - outputs.size());
            total_score += score;
        }

        total_score += net.getTotalNumberOfParameters();
        return total_score;
    }

    private static class InputGenerator implements DoubleSupplier {
        int input_num = 0;

        @Override
        public double getAsDouble() {
            return input_num;
        }
    }

}
