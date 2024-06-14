package com.lucasbrown.GraphNetwork.Local.Nodes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Outcome;
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

    private double bias_gradient;
    private double[] weights_gradient;

    public SimpleNode(final GraphNetwork network, final ActivationFunction activationFunction,
        ITrainableDistribution outputDistribution, IExpectationAdjuster outputAdjuster, 
        ITrainableDistribution signalChanceDistribution, IExpectationAdjuster chanceAdjuster) {
        super(network, activationFunction, outputDistribution, outputAdjuster, signalChanceDistribution, chanceAdjuster);
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
        // weights_delta[i] = new double();
        // }
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
    public void setWeights(int binStr, double[] newWeights){
        weights = newWeights;
    }

    @Override
    public void setBias(int binStr, double newBias){
        bias = newBias;
    }

    @Override
    public int getNumberOfVariables(){
        return weights.length + 1;
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
    public void applyErrorSignals(double epsilon, List<ArrayList<Outcome>> allOutcomes) {
        computeGradient(allOutcomes);
        applyGradient(epsilon);
    }

    private void computeGradient(List<ArrayList<Outcome>> allOutcomes) {
        int T = 0;

        // for all time steps
        for (ArrayList<Outcome> outcomesAtTime : allOutcomes) {
            
            // Compute the probability volume of this timestep
            double probabilityVolume = 1;
            boolean atLeastOnePass = false;
            for (Outcome outcome : outcomesAtTime) {
                probabilityVolume += outcome.probability;
                atLeastOnePass |= outcome.passRate.hasValues();
            }

            // at least one outcome must have a chance to pass through 
            if(!atLeastOnePass){
                continue;
            }

            // Increase the number of non-zero timesteps
            T++;

            // if zero volume, move on to next set
            if (probabilityVolume == 0) {
                continue;
            }

            // add error to the gradient
            for (Outcome outcome : outcomesAtTime) {
                if(!outcome.passRate.hasValues() || outcome.errorDerivative.getProdSum() == 0){
                    continue;
                }
                
                int key = outcome.binary_string;

                double error = outcome.probability * outcome.errorDerivative.getAverage() / probabilityVolume;
                assert Double.isFinite(error);
                bias_gradient += error;

                int i = 0;
                while (key > 0) {
                    if ((key & 0b1) == 1) {
                        weights_gradient[i] += error * outcome.sourceOutcomes[i].netValue;
                        i++;
                    }
                    key = key >> 0b1;
                }
            }
        }//:D heehee ~ Jess hi :) i love you 
        
        //assert T > 0;

        if(T == 0){
            return;
        }

        // test for approximate error
        //double foo = allOutcomes.stream().flatMap(outcomeList -> outcomeList.stream()).mapToDouble(outcome -> outcome.errorOfOutcome.getProdSum()).average().getAsDouble();

        // divide all gradients by the number of non-empty timesteps
        bias_gradient /= T;
        assert Double.isFinite(bias_gradient);

        for (int i = 0; i < weights.length; i++) {
            weights_gradient[i] /= T;
            assert Double.isFinite(weights_gradient[i]);
        }
    }

    private void applyGradient(double epsilon) {
        bias -= bias_gradient * epsilon;
        bias_gradient = 0;

        for (int i = 0; i < weights.length; i++) {
            weights[i] -= weights_gradient[i] * epsilon;
            weights_gradient[i] = 0;
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

}
