package com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators;

import java.util.Arrays;
import java.util.Random;

public class ComplexCombinator extends AdditiveValueCombinator {

    protected double[][] weights;
    protected double[] biases;

    protected int numWeights = 0;

    private Random rng;

    public ComplexCombinator(){
        this(new Random());
    }

    public ComplexCombinator(Random random){
        rng = random;
        weights = new double[1][1];
        biases = new double[1];
        weights[0] = new double[0];
    }

    @Override
    public void notifyNewIncomingConnection() {
        appendWeightsAndBiases();
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
        assert newWeights.length == weights[bitStr].length;
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


    @Override
    public void applyDelta(double[] gradient) {
        if (weights.length == 1) { 
            return; // no connections and no gradient
        }

        int i = 0;
        int key = 1;
        int linear_index = 0;
        while (key < weights.length) {
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

    @Override
    public double[] getLinearizedVariables() {
        double[] vars = new double[getNumberOfVariables()];
        int weightCount = 0;
        for(int key = 1; key < biases.length; key++){
            double[] weight = weights[key];
            System.arraycopy(weight, 0, vars, weightCount, weight.length);
            weightCount += weight.length;
        }

        System.arraycopy(biases, 0, vars, numWeights, biases.length-1);
        return vars;
    }

    @Override
    public void setLinearizedVariables(double[] variables) {
        int weightCount = 0;
        for(int key = 1; key < biases.length; key++){
            double[] weight = weights[key];
            System.arraycopy(variables, weightCount, weight, 0, weight.length);
            weightCount += weight.length;
        }

        System.arraycopy(variables, numWeights, biases, 0, biases.length-1);
    }

    @Override
    public void setLinearizedVariable(int index, double value) {
        if(index >= numWeights){
            biases[index - numWeights] = value;
        }
        else
        {
        // TODO: closed-form solution??? 
        int weightCount = 0;
            for(int key = 1; key < biases.length; key++){
                double[] weight = weights[key];
                if(index < weightCount + weight.length){
                    weight[index - weightCount] = value;
                    return;
                }
                weightCount += weight.length;
            }
        }
    }

}
