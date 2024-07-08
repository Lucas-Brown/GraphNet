package com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators;

import java.util.Collection;

import com.lucasbrown.GraphNetwork.Local.Signal;

public interface IValueCombinator {

    /**
     * Notify the combinator that the node has a new incoming connection
     */
    void notifyNewIncomingConnection();

    /**
     * Get the weights associated with a particular input combination 
     * @param bitStr the binary string corresponding to the input combination
     * @return
     */
    double[] getWeights(int bitStr);

    /**
     * Get the bias associated with a particular input combination 
     * @param bitStr the binary string corresponding to the input combination
     * @return
     */
    double getBias(int bitStr);

    /**
     * Set the weights associated with a particular input combination 
     * @param bitStr the binary string corresponding to the input combination
     */
    void setWeights(int binStr, double[] newWeights);

    /**
     * Set the bias associated with a particular input combination 
     * @param bitStr the binary string corresponding to the input combination
     */
    void setBias(int binStr, double newBias);

    /**
     * Compute the merged signal strengthof a set of incoming signals
     * 
     * @param binary_string the binary string corresponding to the input combination
     * @param incomingSignals The ordered signals sent by the input combination
     * @return The net value resulting from combining the incoming signals
     */
    double computeMergedSignalStrength(Collection<Signal> incomingSignals, int binary_string);

}