package com.lucasbrown.NetworkTraining.GeneticAlgorithm;

import java.util.Arrays;
import java.util.Random;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Local.FilterDistribution;
import com.lucasbrown.GraphNetwork.Local.DataStructure.DataNode;
import com.lucasbrown.GraphNetwork.Local.DataStructure.DataNode.WeightsAndBias;

public class GeneticTrainer {

    private final Random rng = new Random();
    public int popSize;
    public int tournamentSize;
    public IGeneticTrainable[] population;
    public ToDoubleFunction<IGeneticTrainable> fitnessFunction;

    public double new_node_chance = 0.02;
    public double remove_node_chance = 0.01;
    public double new_connection_chance = 0.02;
    public double remove_connection_chance = 0.05;
    public double mutation_rate = 0.5;
    public double maximum_mutation = 0.2;

    public GeneticTrainer(int popSize, int tournamentSize, ToDoubleFunction<IGeneticTrainable> fitnessFunction) {
        this.popSize = popSize;
        this.tournamentSize = tournamentSize;
        this.fitnessFunction = fitnessFunction;

        population = new IGeneticTrainable[popSize];
    }

    public void populateRandom(IGeneticTrainable like, ActivationFunction activationFunction) {

        population[0] = like;
        for (int i = 1; i < popSize; i++) {
            population[i] = like.copy();
            mutate(population[i], 1); // guarantee a mutations for large variance in parameters
        }
    }

    public void getNextPopulation() {

        // create all the children to test
        IGeneticTrainable[] tournament = Arrays.copyOf(population, tournamentSize);
        
        // select two random parents to create a child
        IntStream.range(popSize, tournamentSize)
                .parallel()
                .forEach(i -> tournament[i] = createChild(tournament[rng.nextInt(popSize)], tournament[rng.nextInt(popSize)]));

        // soullessly asses the abilities of each child
        Fitness[] fitness = computeFitness(tournament);

        // keep only the survivors
        population = Stream.of(fitness)
                .map(fit -> fit.individual)
                .limit(popSize)
                .toArray(IGeneticTrainable[]::new);
    }

    public IGeneticTrainable createChild(IGeneticTrainable parent1, IGeneticTrainable parent2) {
        // create a child and attempt to cross it with its parent
        IGeneticTrainable child = parent1.copy();
        crossOver(child, parent2);

        // mutate parameters
        mutate(child); // hehe

        // attempt structural changes
        attemptStructureChanges(child);

        return child;
    }

    public Fitness[] computeFitness(IGeneticTrainable[] tournament) {
        return IntStream.range(0, tournament.length)
                //.parallel()
                .mapToObj(i -> new Fitness(tournament[i], fitnessFunction.applyAsDouble(tournament[i])))
                .sorted()
                .toArray(Fitness[]::new);
    }

    public void mutate(IGeneticTrainable toMutate, double mutation_rate) {
        for (int n = 0; n < toMutate.getNumberOfNodes(); n++) {
            DataNode node = toMutate.getNode(n);

            for (int bin_str = 1; bin_str <= node.getNumberOfBiases(); bin_str++) {
                WeightsAndBias wAb = node.getWeightsAndBias(bin_str);
                FilterDistribution[] transferFunctions = node.getFilters();

                // mutate bias
                if (rng.nextDouble() < mutation_rate)
                    wAb.bias *= relativeMutation();

                // mutate weights
                for (int i = 0; i < wAb.weights.length; i++) {
                    if (rng.nextDouble() < mutation_rate)
                        wAb.weights[i] *= relativeMutation();
                }

                // mutate transfer function parameters
                for (FilterDistribution dist : transferFunctions) {
                    double[] params = dist.getParameters();
                    for (int i = 0; i < params.length; i++) {
                        if (rng.nextDouble() < mutation_rate)
                            params[i] *= relativeMutation();
                    }
                    dist.setParameters(params);
                }

            }
        }
    }

    public void mutate(IGeneticTrainable toMutate) {
        mutate(toMutate, this.mutation_rate);
    }

    private double relativeMutation() {
        return rng.nextDouble() * 2 * maximum_mutation - maximum_mutation + 1;
    }

    /**
     * apply a single cross-over of 'genes' from parent 2 to parent 1 ( i.e, parent
     * 1 will be altered)
     * 
     * @param parent1
     * @param parent2
     */
    public boolean crossOver(IGeneticTrainable parent1, IGeneticTrainable parent2) {
        if (!parent1.isCompatibleWith(parent2))
            return false;

        final int n_nodes = parent1.getNumberOfNodes();
        int crossover_idx = rng.nextInt(n_nodes);

        DataNode node1 = parent1.getNode(crossover_idx);
        DataNode node2 = parent2.getNode(crossover_idx);
        int crossover_sub_idx = rng.nextInt(node1.getTotalNumberOfParameters());

        // sub-index cross-over
        crossOverPartial(node1, node2, crossover_sub_idx);

        // remaining cross-overs
        for (crossover_idx++; crossover_idx < n_nodes; crossover_idx++) {
            node1 = parent1.getNode(crossover_idx);
            node2 = parent2.getNode(crossover_idx);
            crossOverFull(node1, node2);
        }

        return true;
    }

    public void crossOverPartial(DataNode node1, DataNode node2, int start_idx) {
        int idx = 0;
        int bin_str = 1;
        int parameter_n = 0;

        if (start_idx < node1.getNumberOfWeightAndBiasParameters()) {
            // count up to the start index
            // doing this analytically is impossible as we'd need the Lambdert-W function
            // this could be more efficient but this is easier to understand

            while (idx < start_idx) {
                parameter_n++;
                idx++;
                if (parameter_n > node1.getWeightsAndBias(bin_str).weights.length + 1) {
                    bin_str++;
                    parameter_n = 0;
                }

            }

            // cross over the partial weights and bias
            WeightsAndBias wAb1 = node1.getWeightsAndBias(bin_str);
            WeightsAndBias wAb2 = node2.getWeightsAndBias(bin_str);
            int weight_length = wAb1.weights.length;

            for (; parameter_n < weight_length; parameter_n++) {
                wAb1.weights[parameter_n] = wAb2.weights[parameter_n];
            }

            wAb1.bias = wAb2.bias;

            idx += weight_length - parameter_n; // - 1 + 1

            // cross over full weights and biases
            for (bin_str++; bin_str < node1.getNumberOfBiases(); bin_str++) {
                wAb1 = node1.getWeightsAndBias(bin_str);
                wAb2 = node2.getWeightsAndBias(bin_str);
                weight_length = wAb1.weights.length;

                for (parameter_n = 0; parameter_n < weight_length; parameter_n++) {
                    wAb1.weights[parameter_n] = wAb2.weights[parameter_n];
                }

                wAb1.bias = wAb2.bias;
                idx += weight_length;
            }
        }

        start_idx -= node1.getNumberOfWeightAndBiasParameters();
        int filter_n = 0;
        idx = 0;
        FilterDistribution[] filters1 = node1.getFilters();
        FilterDistribution[] filters2 = node2.getFilters();

        if (start_idx > 0) {
            while (idx < start_idx) {
                if (idx > filters1[filter_n].getParameters().length) {
                    filter_n++;
                    idx = 0;
                } else {
                    idx++;
                }
            }

            // cross over partial filters
            double[] filter_data = filters1[filter_n].getParameters().clone();
            double[] filter_cross = filters2[filter_n].getParameters();
            for (; idx < filter_data.length; idx++) {
                filter_data[idx] = filter_cross[idx];
            }
            filters1[filter_n].setParameters(filter_data);
            filter_n++;
        }

        // cross over full filters
        for (; filter_n < filters1.length; filter_n++) {
            filters1[filter_n].setParameters(filters2[filter_n].getParameters());
        }

    }

    public void crossOverFull(DataNode node1, DataNode node2) {
        // copy weights and biases
        for (int i = 1; i < node1.getNumberOfBiases(); i++) {
            WeightsAndBias wAb1 = node1.getWeightsAndBias(i);
            WeightsAndBias wAb2 = node2.getWeightsAndBias(i);
            wAb1.bias = wAb2.bias;
            wAb1.weights = wAb2.weights.clone();
        }

        // copy transfer function parameters
        FilterDistribution[] filters1 = node1.getFilters();
        FilterDistribution[] filters2 = node2.getFilters();
        for (int i = 0; i < filters1.length; i++) {
            filters1[i].setParameters(filters2[i].getParameters());
        }
    }

    private void attemptStructureChanges(IGeneticTrainable individual) {
        attemptNodeChange(individual);
        attemptConnectionChange(individual);
    }

    private void attemptNodeChange(IGeneticTrainable individual) {
        if (rng.nextDouble() < new_node_chance) {
            addNewNode(individual);
        }

        if (rng.nextDouble() < remove_node_chance) {
            int n_nodes = individual.getNumberOfNodes();
            int minimum = individual.getMinimumNumberOfNodes();
            removeNode(individual, rng.nextInt(n_nodes - minimum) + minimum);
        }
    }

    private void addNewNode(IGeneticTrainable individual) {
        // TODO: remove hard-coding of activation function
        int id = individual.addNewNode(ActivationFunction.LINEAR);
 
        // add a new incoming and outgoing connection to make the node relevant
        addNewConnection(individual, id, getRandomIncomingConnection(individual)); // outgoing
        addNewConnection(individual, getRandomOutgoingConnection(individual), id); // incoming
    }

    private void removeNode(IGeneticTrainable individual, int node_idx) {
        individual.removeNode(node_idx);
    }

    private int getRandomIncomingConnection(IGeneticTrainable individual){
        int total = individual.getNumberOfNodes();
        int n_input = individual.getNumberOfInputNodes();
        return rng.nextInt(total-n_input)+n_input;
    }

    private int getRandomOutgoingConnection(IGeneticTrainable individual){
        int total = individual.getNumberOfNodes();
        int n_input = individual.getNumberOfInputNodes();
        int n_output = individual.getNumberOfOutputNodes();
        int roll = rng.nextInt(total - n_output); // avoid output region
        roll += roll < n_input ? 0 : n_output + n_input -1;
        return roll;
    }

    private void attemptConnectionChange(IGeneticTrainable individual) {

        if (rng.nextDouble() < new_connection_chance) {
            addNewConnection(individual, getRandomOutgoingConnection(individual), getRandomIncomingConnection(individual)); // two random nodes
        }

        if (rng.nextDouble() < remove_connection_chance) {
            int node_id = rng.nextInt(individual.getNumberOfNodes());
            DataNode node = individual.getNode(node_id);

            // randomly select a connection to remove
            int[] incoming_ids = node.getIncomingConnectionIDs();
            int[] outgoing_ids = node.getIncomingConnectionIDs();
            if(incoming_ids.length + outgoing_ids.length == 0)
                return;

            int remove_idx = rng.nextInt(incoming_ids.length + outgoing_ids.length);
            if (remove_idx < incoming_ids.length) {
                // remove and incoming connection
                removeConnection(individual, incoming_ids[remove_idx], node_id);
            } else {
                // remove outgoing connection
                removeConnection(individual, node_id, outgoing_ids[remove_idx - incoming_ids.length]);
            }

        }
    }

    private void addNewConnection(IGeneticTrainable individual, int from_id, int to_id) {
        // TODO: remove hard-coding of transfer-function
        individual.addNewConnection(from_id, to_id,
                new BellCurveDistribution(rng.nextGaussian(), rng.nextDouble() + 0.5));
    }

    public void removeConnection(IGeneticTrainable individial, int from_id, int to_id) {
        individial.removeConnection(from_id, to_id);
    }

    private class Fitness implements Comparable<Fitness> {
        private final IGeneticTrainable individual;
        private final double fitness;

        private Fitness(IGeneticTrainable trainable, double fitness) {
            this.individual = trainable;
            this.fitness = fitness;
        }

        @Override
        public int compareTo(Fitness o) {
            return Double.compare(fitness, o.fitness);
        }
    }

}
