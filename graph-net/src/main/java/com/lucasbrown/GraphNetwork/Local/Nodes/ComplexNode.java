package com.lucasbrown.GraphNetwork.Local.Nodes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.stream.IntStream;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Signal;

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of
 * sending a signal out to other connected nodes
 */
public class ComplexNode extends NodeBase {

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

    public ComplexNode(final ActivationFunction activationFunction){
        this(null, activationFunction);
    }

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
    }

    /**
     * Add an incoming connection to the node
     * 
     * @param connection
     * @return true
     */
    @Override
    public boolean addIncomingConnection(Arc connection) {
        appendWeightsAndBiases();
        return super.addIncomingConnection(connection);
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

    /**
     * Adds another layer of depth to the weights and biases hyper array
     */
    private void appendWeightsAndBiases() {
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
            biases[i] = rng.nextDouble();

            // populate the weights array
            int count = weights[i - old_size].length + 1;
            weights[i] = new double[count];
            for (int j = 0; j < count; j++) {
                weights[i][j] = rng.nextDouble();
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

        ArrayList<Signal> sortedSignals = sortSignalByID(incomingSignals);

        double[] input_weights = weights[binary_string];

        double strength = IntStream.range(0, input_weights.length)
                .mapToDouble(i -> input_weights[i] * sortedSignals.get(i).strength)
                .sum();

        strength += biases[binary_string];

        return strength;
    }

    @Override
    public void applyErrorSignals(double epsilon) {
        for (int key = 1; key < biases.length; key++) {
            int count = delta_counts[key];
            if (count == 0)
                continue;

            double delta = -epsilon / count;
            biases[key] += delta * bias_delta[key];
            bias_delta[key] = 0;

            for (int i = 0; i < weights[key].length; i++) {
                weights[key][i] += delta * weights_delta[key][i];
                weights_delta[key][i] = 0;
            }

            delta_counts[key] = 0;
        }

        for (Arc connection : outgoing) {
            connection.probDist.applyAdjustments();
        }

        error_signals.clear();
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
        delta_counts[bitStr] += 1;
    }
    
    @Override
    protected void addWeightDelta(int bitStr, int weight_index, double error){
        weights_delta[bitStr][weight_index] += error; 
    }
    
    public static InputNode asInputNode(ActivationFunction activator){
        return new InputNode(new ComplexNode(activator));
    }
    
    public static OutputNode asOutputNode(ActivationFunction activator){
        return new OutputNode(new ComplexNode(activator));
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
