package com.lucasbrown.GraphNetwork.Local.ComplexNode;

import java.security.InvalidAlgorithmParameterException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.lucasbrown.GraphNetwork.Distributions.FilterDistribution;
import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.SharedNetworkData;
import com.lucasbrown.NetworkTraining.ApproximationTools.ArrayTools;
import com.lucasbrown.NetworkTraining.ApproximationTools.Convolution.FilterDistributionConvolution;

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of
 * sending a signal out to other connected nodes
 */
public class ComplexNode extends Node {

    /**
     * Maps all incoming node ID's to an int from 0 to the number of incoming nodes
     * -1
     */
    protected final HashMap<Integer, Integer> orderedIDMap;

    /**
     * Each possible combinations of inputs has a corresponding unique set of
     * weights and biases
     * both grow exponentially, which is bad, but every node should have relatively
     * few connections
     */
    protected double[][] weights;
    protected double[] biases;

    /**
     * 
     */
    private HashMap<TimeKey, Double> error_signals;

    private int[] delta_counts;
    private double[] bias_delta;
    private double[][] weights_delta;

    /**
     * The binary string representation of the incoming arcs being sent a backwards
     * signal
     */
    private int backwardsBinStr;

    protected boolean hasValidForwardSignal;

    public ComplexNode(final GraphNetwork network, final ActivationFunction activationFunction) {
        super(network, activationFunction);
        weights = new double[1][1];
        biases = new double[1];
        weights[0] = new double[0];
        delta_counts = new int[1];
        bias_delta = new double[1];
        weights_delta = new double[1][1];
        weights_delta[0] = new double[0];
        error_signals = new HashMap<>();

        uniqueIncomingNodeIDs = new HashSet<>();
    }

    /**
     * Add an incoming connection to the node
     * 
     * @param connection
     * @return true
     */
    @Override
    public boolean addIncomingConnection(Arc connection) {
        orderedIDMap.put(connection.sending.id, 1 << orderedIDMap.size());
        appendWeightsAndBiases();
        return super.addIncomingConnection(connection);
    }

    /**
     * Notify this node of a new incoming forward signal
     * 
     * @param signal
     */
    void recieveForwardSignal(Signal signal) {
        appendForward(signal);
        super.recieveForwardSignal(signal);
    }

    @Override
    public void recieveError(int timestep, int key, double error) {
        TimeKey tk = new TimeKey(timestep, key);
        Double error_rate = error_signals.get(tk);
        if (error_rate == null) {
            error_rate = Double.valueOf(0);
        }
        error_rate += error;
        error_signals.put(tk, error_rate);
    }

    @Override
    public Double getError(int timestep, int key) {
        return error_signals.get(new TimeKey(timestep, key));
    }

    private void appendForward(Signal signal) {
        int signal_id = orderedIDMap.get(signal.sendingNode.id);
        ArrayList<Signal> signals = forwardNext.get(signal_id);
        if (signals == null) {
            signals = new ArrayList<Signal>(1);
            signals.add(signal);
            forwardNext.put(signal_id, signals);
        } else {
            signals.add(signal);
        }
        uniqueIncomingNodeIDs.add(signal.sendingNode.id);
    }

    /**
     * Adds another layer of depth to the weights and biases hyper array
     */
    private void appendWeightsAndBiases() {
        Random rand = new Random();
        final int old_size = biases.length;
        final int new_size = old_size * 2;

        // the first half doesn't need to be changed
        biases = Arrays.copyOf(biases, new_size);
        weights = Arrays.copyOf(weights, new_size);

        delta_counts = new int[new_size];
        bias_delta = new double[new_size];
        weights_delta = new double[new_size][];

        // the second half needs entirely new data
        for (int i = old_size; i < new_size; i++) {
            biases[i] = rand.nextDouble();

            // populate the weights array
            int count = weights[i - old_size].length + 1;
            weights[i] = new double[count];
            for (int j = 0; j < count; j++) {
                weights[i][j] = rand.nextDouble();
            }
        }

        for (int i = 0; i < new_size; i++) {
            weights_delta[i] = new double[weights.length];
        }
    }

    @Override
    public double[] getWeights(int bitStr) {
        return weights[bitStr].clone(); // A shallow clone is okay here
    }

    @Override
    public double getBias(int bitStr){
        return biases[bitStr];
    }
    
    /**
     * Compute the merged signal strength of a set of incoming signals
     * 
     * @param incomingSignals
     * @return
     */
    @Override
    protected double computeMergedSignalStrength(Collection<Signal> incomingSignals, int binary_string) {

        ArrayList<Signal> sortedSignals = new ArrayList<>(incomingSignals);
        // sorting by id to ensure that the weights are applied to the correct
        // node/signal
        sortedSignals.sort((s1, s2) -> Integer.compare(s1.recievingNode.id, s2.recievingNode.id));

        double[] input_weights = weights[binary_string];

        double strength = IntStream.range(0, input_weights.length)
                .mapToDouble(i -> input_weights[i] * sortedSignals.get(i).strength)
                .sum();

        strength += biases[binary_string];

        return strength;
    }

    /**
     * Attempt to send forward and backward signals
     */
    public void sendTrainingSignals() {

        if (!outgoing.isEmpty()) {
            // Send the forward signals and record the cumulative error
            sendForwardSignals();
            hasValidForwardSignal = false;
        }

        /*
         * if (!incoming.isEmpty()) {
         * sendBackwardsSignals();
         * }
         */

    }

    @Override
    protected void addBiasDelta(int bitStr, double error){
        bias_delta[bitStr] += error;
        delta_counts[binary_string] += 1;
    }
    
    @Override
    protected void addWeightDelta(int bitStr, int weight_index, double error){
        weights_delta[bitStr][weight_index] += error; 
    }
    
    private class TimeKey {
        private final int timestep;
        private final int key;

        public TimeKey(int timestep, int key) {
            this.timestep = timestep;
            this.key = key;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof TimeKey))
                return false;

            TimeKey tk = (TimeKey) o;
            return key == tk.key & timestep == tk.timestep;
        }

        @Override
        public int hashCode() {
            return timestep << 16 + key;
        }
    }
}
