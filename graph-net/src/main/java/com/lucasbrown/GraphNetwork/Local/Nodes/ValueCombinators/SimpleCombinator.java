package com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators;

import java.util.Arrays;
import java.util.Random;

import com.lucasbrown.HelperClasses.IterableTools;

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of
 * sending a signal out to other connected nodes
 */
public class SimpleCombinator extends AdditiveValueCombinator {

    private Random rng;
    protected double[] weights;
    protected double bias;

    public SimpleCombinator() {
        this(new Random());
    }

    public SimpleCombinator(Random random) {
        rng = random;
        weights = new double[0];
        bias = rng.nextGaussian();
    }

    @Override
    public void notifyNewIncomingConnection() {
        appendWeights();
    }

    /**
     * Adds another layer of depth to the weights and biases hyper array
     */
    private void appendWeights() {
        weights = Arrays.copyOf(weights, weights.length + 1);
        weights[weights.length - 1] = rng.nextGaussian();
    }

    @Override
    public double[] getWeights(int bitStr) {
        return IterableTools.applyMask(weights, bitStr);
    }

    @Override
    public double getBias(int bitStr) {
        return bias;
    }

    @Override
    public void setWeights(int binStr, double[] newWeights) {
        weights = newWeights;
    }

    @Override
    public void setBias(int binStr, double newBias) {
        bias = newBias;
    }

    @Override
    public int getNumberOfVariables() {
        return weights.length == 0 ? 0 : weights.length + 1;
    }

    @Override
    public int getLinearIndexOfWeight(int key, int weight_index) {
        int lin_idx = 0;
        while (key > 0) {
            if ((key & 0b1) == 1) {
                weight_index--;
            }
            if (weight_index < 0) {
                return lin_idx;
            }
            lin_idx++;
            key = key >> 0b1;
        }
        return -1;
    }

    @Override
    public int getLinearIndexOfBias(int key) {
        return weights.length;
    }

    @Override
    public void applyDelta(double[] gradient) {
        if (weights.length == 0) { 
            return; // no connections and no gradient
        }
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= gradient[i];
            gradient[i] = 0;
        }

        bias -= gradient[gradient.length - 1];
    }

}
