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

    private double bias_gradient;
    private double[] weights_gradient;

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
        bias_gradient = 0;
        weights_gradient = new double[0];
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
        weights_gradient = new double[weights.length];
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
    public void applyErrorSignals(double epsilon, List<ArrayList<Outcome>> allOutcomes) {
        computeGradient(allOutcomes);
        applyGradient(epsilon);
    }
    
    private void computeGradient(List<ArrayList<Outcome>> allOutcomes){
        // for all time steps
        for (ArrayList<Outcome> outcomesAtTime : allOutcomes) {
            
            // Compute the probability volume of this timestep 
            double probabilityVolume = 0;
            for(Outcome outcome : outcomesAtTime){
                probabilityVolume += outcome.probability;
            }

            // if zero volume, move on to next set
            if(probabilityVolume == 0){
                continue;
            }

            // add error to the gradient
            for(Outcome outcome : outcomesAtTime){
                int key = outcome.binary_string;

                double error = outcome.errorOfOutcome.getProdSum()/probabilityVolume;
                assert Double.isFinite(error);
                bias_gradient += error;

                int i = 0;
                while (key > 0) {
                    if((key & 0b1) == 1)
                    {
                        weights_gradient[i] += error * outcome.sourceOutcomes[i].activatedValue;
                        i++;
                    }
                    key = key >> 0b1;
                }
            }
        }

        // divide all gradients by the number of non-empty timesteps 
        int T = allOutcomes.size();
        bias_gradient /= T;

        for (int i = 0; i < weights.length; i++) {
            weights_gradient[i] /= T;
        }
    }

    private void applyGradient(double epsilon){
        bias -= bias_gradient * epsilon;
        bias_gradient = 0;

        for (int i = 0; i < weights.length; i++) {
            weights[i] -= weights_gradient[i] * epsilon;
            weights_gradient[i] = 0;
        }

        for (Arc connection : outgoing) {
            connection.probDist.applyAdjustments();
        }
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
