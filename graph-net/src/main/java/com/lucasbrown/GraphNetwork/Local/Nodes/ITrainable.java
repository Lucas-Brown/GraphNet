package com.lucasbrown.GraphNetwork.Local.Nodes;

import java.util.ArrayList;
import java.util.List;

import com.lucasbrown.GraphNetwork.Local.Outcome;
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
     * @return
     */
    public int getNumberOfVariables();

    public int getIndexFromKey(int key);

    /**
     * Get the distribution of values that are output by this node 
     * @return
     */
    public ITrainableDistribution getOutputDistribution();

    /**
     * Get the distribution of how likely an output from this node is going to become an output.
     * @return
     */
    public ITrainableDistribution getSignalChanceDistribution();

    
    public IExpectationAdjuster getOutputDistributionAdjuster();

    public IExpectationAdjuster getSignalChanceDistributionAdjuster();

    public void applyErrorSignals(double epsilon, List<ArrayList<Outcome>> allOutcomes);

    public void applyDistributionUpdate();

    public void applyFilterUpdate();

}
