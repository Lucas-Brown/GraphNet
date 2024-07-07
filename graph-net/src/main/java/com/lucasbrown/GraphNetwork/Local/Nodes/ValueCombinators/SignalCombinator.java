package com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators;

import java.util.Collection;
import java.util.Iterator;
import java.util.NoSuchElementException;

import com.lucasbrown.GraphNetwork.Local.Signal;

public abstract class SignalCombinator {
    
    /**
     * Notify the combinator that the node has a new incoming connection
     */
    public abstract void notifyNewIncomingConnection();

    /**
     * Get the weights associated with a particular input combination 
     * @param bitStr the binary string corresponding to the input combination
     * @return
     */
    public abstract double[] getWeights(int bitStr);

    /**
     * Get the bias associated with a particular input combination 
     * @param bitStr the binary string corresponding to the input combination
     * @return
     */
    public abstract double getBias(int bitStr);
    
    /**
     * Set the weights associated with a particular input combination 
     * @param bitStr the binary string corresponding to the input combination
     */
    public abstract void setWeights(int binStr, double[] newWeights);
    
    /**
     * Set the bias associated with a particular input combination 
     * @param bitStr the binary string corresponding to the input combination
     */
    public abstract void setBias(int binStr, double newBias);

    /**
     * @return The total number of variables (weights + biases) 
     */
    public abstract int getNumberOfVariables();

    /**
     * Compute the merged signal strengthof a set of incoming signals
     * 
     * @param binary_string the binary string corresponding to the input combination
     * @param incomingSignals The ordered signals sent by the input combination
     * @return The net value resulting from combining the incoming signals
     */
    public double computeMergedSignalStrength(Collection<Signal> incomingSignals, int binary_string) {

        double strength = getBias(binary_string);
        double[] weights_of_signals = getWeights(binary_string);

        Iterator<Signal> signalIterator = incomingSignals.iterator();
        try{
            for (int i = 0; i < weights_of_signals.length; i++) {
                strength += signalIterator.next().getOutputStrength() * weights_of_signals[i];
            }
        } catch(NoSuchElementException e){
            // check to see if the cause is the binary string or the weights
            int binCount = Integer.bitCount(binary_string);
            if(binCount != weights_of_signals.length){
                throw new CombinatorMissalignmentException("Number of weights does not equal the number of input combinations indicated by the binary string.", e);
            }
            else if(binCount != incomingSignals.size()){
                throw new CombinatorMissalignmentException("Size of the incoming signal set is not equal the number of input combinations indicated by the binary string. Please ensure that the binary string is proper.", e);
            }
            else{
                throw new CombinatorMissalignmentException("Inconclusive missalignment: " + e.getMessage(), e);
            }
        }
        

        assert Double.isFinite(strength);
        return strength;
    }
}
