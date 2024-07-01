package com.lucasbrown.GraphNetwork.Local.Nodes;

import com.lucasbrown.NetworkTraining.DataSetTraining.IExpectationAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.ITrainableDistribution;

public interface ITrainable extends INode {

    // getters/setters for weights and bias
    public double[] getWeights(int bitStr);

    public double getBias(int bitStr);

    public void setWeights(int bitStr, double[] newWeights);

    public void setBias(int bitStr, double newBias);

    /**
     * Get the total number of weights and biases combined
     * 
     * @return
     */
    public int getNumberOfVariables();

    
    /**
     * Get the total number of incoming distribution parameters
     * 
     * @return
     */
    public int getNumberOfParameters();

    /**
     * Returns the unique index of the key-weight pair
     * 
     * @param key
     * @param weight_index
     * @return
     */
    public int getLinearIndexOfWeight(int key, int weight_index);

    /**
     * Returns the unique index of the bias given this key
     * 
     * @param key
     * @param weight_index
     * @return
     */
    public int getLinearIndexOfBias(int key);

    /**
     * Get the distribution of values that are output by this node
     * 
     * @return
     */
    public ITrainableDistribution getOutputDistribution();

    /**
     * Get the distribution of how likely an output from this node is going to
     * become an output.
     * 
     * @return
     */
    public ITrainableDistribution getSignalChanceDistribution();

    public IExpectationAdjuster getOutputDistributionAdjuster();

    public IExpectationAdjuster getSignalChanceDistributionAdjuster();

    public void applyDistributionUpdate();

    public void applyFilterUpdate();

    public void applyDelta(double[] gradient);

    public void setParameters(double[] params);

}
