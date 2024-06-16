package com.lucasbrown.GraphNetwork.Local.Nodes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Signal;
import com.lucasbrown.NetworkTraining.ApproximationTools.IterableTools;
import com.lucasbrown.NetworkTraining.DataSetTraining.IExpectationAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.ITrainableDistribution;

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of
 * sending a signal out to other connected nodes
 */
public class SimpleNode extends TrainableNodeBase {

    protected double[] weights;
    protected double bias;

    public SimpleNode(final GraphNetwork network, final ActivationFunction activationFunction,
            ITrainableDistribution outputDistribution, IExpectationAdjuster outputAdjuster,
            ITrainableDistribution signalChanceDistribution, IExpectationAdjuster chanceAdjuster) {
        super(network, activationFunction, outputDistribution, outputAdjuster, signalChanceDistribution,
                chanceAdjuster);
        weights = new double[0];
        bias = rng.nextGaussian();
    }

    /**
     * Add an incoming connection to the node
     * 
     * @param connection
     * @return true
     */
    @Override
    public boolean addIncomingConnection(Arc connection) {
        appendWeights();
        return super.addIncomingConnection(connection);
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
        return weights.length + 1;
    }

    @Override
    public int getLinearIndexOfWeight(int key, int weight_index) {
        return weight_index;
    }

    @Override
    public int getLinearIndexOfBias(int key) {
        return weights.length;
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

        double strength = bias;
        double[] weights_of_signals = getWeights(binary_string);

        for (int i = 0; i < weights_of_signals.length; i++) {
            strength += sortedSignals.get(i).getOutputStrength() * weights_of_signals[i];
        }

        assert Double.isFinite(strength);
        return strength;
    }

    @Override
    public void applyGradient(double[] gradient, double epsilon) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= gradient[i] * epsilon;
            gradient[i] = 0;
        }

        bias -= gradient[gradient.length - 1] * epsilon;
    }

}
