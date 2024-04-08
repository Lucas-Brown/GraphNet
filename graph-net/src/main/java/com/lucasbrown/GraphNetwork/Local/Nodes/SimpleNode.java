package com.lucasbrown.GraphNetwork.Local.Nodes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Signal;
import com.lucasbrown.NetworkTraining.ApproximationTools.ArrayTools;

/**
 * A node within a graph neural network.
 * Capable of sending and recieving signals from other nodes.
 * Each node uses a @code NodeConnection to evaluate its own likelyhood of
 * sending a signal out to other connected nodes
 */
public class SimpleNode extends NodeBase {

    protected double[] weights;
    protected double bias;

    /**
     * 
     */
    private HashMap<Integer, Double> error_signals;

    private double probability_weight_sum;
    private double bias_delta;
    private double[] weights_delta;

    /**
     * The binary string representation of the incoming arcs being sent a backwards
     * signal
     */

    protected boolean hasValidForwardSignal;

    public SimpleNode(final ActivationFunction activationFunction){
        this(null, activationFunction);
    }

    public SimpleNode(final GraphNetwork network, final ActivationFunction activationFunction) {
        super(network, activationFunction);
        weights = new double[0];
        bias = rng.nextGaussian();
        probability_weight_sum = 0;
        bias_delta = 0;
        weights_delta = new double[0];
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
        appendWeights();
        return super.addIncomingConnection(connection);
    }

    @Override
    public void recieveError(int timestep, int key, double error) {
        Double error_rate = error_signals.get(timestep);
        if (error_rate == null) {
            error_rate = Double.valueOf(0);
        }
        error_rate += error;
        error_signals.put(timestep, error_rate);
    }

    @Override
    public Double getError(int timestep, int key) {
        return error_signals.get(timestep);
    }

    /**
     * Adds another layer of depth to the weights and biases hyper array
     */
    private void appendWeights() {
        weights = Arrays.copyOf(weights, weights.length + 1);
        weights_delta = new double[weights.length];
        weights[weights.length - 1] = rng.nextGaussian();
    }

    @Override
    public double[] getWeights(int bitStr) {
        return ArrayTools.applyMask(weights, bitStr);
    }

    @Override
    public double getBias(int bitStr){
        return bias;
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

        double strength = bias;
        double[] weights_of_signals = getWeights(binary_string);
         
        for(int i = 0; i < weights_of_signals.length; i++){
            strength += sortedSignals.get(i).strength * weights_of_signals[i];
        }

        return strength;
    }

    @Override
    public void applyErrorSignals(double epsilon) {
        if (probability_weight_sum == 0)
            return;

        double delta = -epsilon / probability_weight_sum;
        bias += delta * bias_delta;
        probability_weight_sum = 0;
        bias_delta = 0;

        for (int i = 0; i < weights.length; i++) {
            weights[i] += delta * weights_delta[i];
            weights_delta[i] = 0;
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
    protected void addProbabilityWeight(int bitStr, double weight){
        probability_weight_sum += weight;
    }

    @Override
    protected void addBiasDelta(int bitStr, double error){
        bias_delta += error;
    }
    
    @Override
    protected void addWeightDelta(int bitStr, int weight_index, double error){
        weights_delta[weight_index] += error; 
    }
    
    public static InputNode asInputNode(ActivationFunction activator){
        return new InputNode(new SimpleNode(activator));
    }
    
    public static OutputNode asOutputNode(ActivationFunction activator){
        return new OutputNode(new SimpleNode(activator));
    } 

}
