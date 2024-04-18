package com.lucasbrown.GraphNetwork.Local.Nodes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Signal;
import com.lucasbrown.NetworkTraining.ApproximationTools.ArrayTools;
import com.lucasbrown.NetworkTraining.ApproximationTools.WeightedAverage;

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

    private double bias_delta;
    private double[] weights_delta;

    /**
     * The binary string representation of the incoming arcs being sent a backwards
     * signal
     */

    public SimpleNode(final ActivationFunction activationFunction){
        this(null, activationFunction);
    }

    public SimpleNode(final GraphNetwork network, final ActivationFunction activationFunction) {
        super(network, activationFunction);
        weights = new double[0];
        bias = rng.nextGaussian();
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

    /**
     * Adds another layer of depth to the weights and biases hyper array
     */
    private void appendWeights() {
        weights = Arrays.copyOf(weights, weights.length + 1);
        weights[weights.length - 1] = rng.nextGaussian();
        weights_delta = new double[weights.length];
        // for (int i = 0; i < weights_delta.length; i++) {
        //     weights_delta[i] = new double();
        // }
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
            strength += sortedSignals.get(i).getOutputStrength() * weights_of_signals[i];
        }

        return strength;
    }

    @Override
    public void applyErrorSignals(double epsilon, HashMap<Integer, ArrayList<Outcome>> allOutcomes) {
        double delta = -epsilon / error_signals.size();
        bias += delta * bias_delta;
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
    
    
    public static InputNode asInputNode(ActivationFunction activator){
        return new InputNode(new SimpleNode(activator));
    }
    
    public static OutputNode asOutputNode(ActivationFunction activator){
        return new OutputNode(new SimpleNode(activator));
    }

}
