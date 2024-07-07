package com.lucasbrown.GraphNetwork.Local.Nodes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.stream.IntStream;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Edge;
import com.lucasbrown.GraphNetwork.Local.Signal;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.IExpectationAdjuster;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.ITrainableDistribution;

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of
 * sending a signal out to other connected nodes
 */
public class ComplexNode extends TrainableNodeBase {

    /**
     * Each possible combinations of inputs has a corresponding unique set of
     * weights and biases
     * both grow exponentially, which is bad, but every node should have relatively
     * few connections
     */
    protected double[][] weights;
    protected double[] biases;

    protected int numWeights = 0;

    public ComplexNode(final GraphNetwork network, final ActivationFunction activationFunction,
            ITrainableDistribution outputDistribution, IExpectationAdjuster outputAdjuster,
            ITrainableDistribution signalChanceDistribution, IExpectationAdjuster chanceAdjuster) {
        super(network, activationFunction, outputDistribution, outputAdjuster, signalChanceDistribution,
                chanceAdjuster);
        weights = new double[1][1];
        biases = new double[1];
        weights[0] = new double[0];
    }

    /**
     * Add an incoming connection to the node
     * 
     * @param connection
     * @return true
     */
    @Override
    public boolean addIncomingConnection(Edge connection) {
        appendWeightsAndBiases();
        return super.addIncomingConnection(connection);
    }

    /**
     * Adds another layer of depth to the weights and biases hyper array
     */
    private void appendWeightsAndBiases() {
        final int old_size = biases.length;
        final int new_size = old_size * 2;

        // the first half doesn't need to be changed
        biases = Arrays.copyOf(biases, new_size);
        weights = Arrays.copyOf(weights, new_size);

        numWeights += numWeights + old_size;

        // the second half needs entirely new data
        for (int i = old_size; i < new_size; i++) {
            biases[i] = rng.nextDouble();

            // populate the weights array
            int count = weights[i - old_size].length + 1;
            weights[i] = new double[count];
            for (int j = 0; j < count; j++) {
                weights[i][j] = rng.nextDouble();
            }
        }
    }

    @Override
    public double[] getWeights(int bitStr) {
        return weights[bitStr].clone(); // A shallow clone is okay here
    }

    @Override
    public double getBias(int bitStr) {
        return biases[bitStr];
    }

    @Override
    public void setWeights(int bitStr, double[] newWeights) {
        weights[bitStr] = newWeights;
    }

    @Override
    public void setBias(int bitStr, double newBias) {
        biases[bitStr] = newBias;
    }

    @Override
    public int getLinearIndexOfWeight(int key, int weight_index) {
        // may be a closed form
        int index = 0;
        for (int i = 1; i < key; i++) {
            index += weights[i].length;
        }
        return index + weight_index;
    }

    @Override
    public int getLinearIndexOfBias(int key) {
        return numWeights + key - 1; // key = 0 aligns with the end of weights
    }

    /**
     * Compute the merged signal strength of a set of incoming signals
     * 
     * @param incomingSignals
     * @return
     */
    @Override
    public double computeMergedSignalStrength(Collection<Signal> incomingSignals, int binary_string) {

        ArrayList<Signal> sortedSignals = sortSignalByID(incomingSignals);

        double[] input_weights = weights[binary_string];

        double strength = IntStream.range(0, input_weights.length)
                .mapToDouble(i -> input_weights[i] * sortedSignals.get(i).getOutputStrength())
                .sum();

        strength += biases[binary_string];

        return strength;
    }

    @Override
    public void applyDelta(double[] gradient) {

        int i = 0;
        int key = 1;
        int linear_index = 0;
        while (key < getNumInputCombinations()) {
            weights[key][i] -= gradient[linear_index];
            i++;
            linear_index++;
            if (i >= weights[key].length) {
                i = 0;
                key++;
            }
        }

        for (key = 1; key < biases.length; key++) {
            biases[key] -= gradient[linear_index++];
        }
    }

    @Override
    public int getNumberOfVariables() {
        return numWeights + biases.length - 1;
    }

}
