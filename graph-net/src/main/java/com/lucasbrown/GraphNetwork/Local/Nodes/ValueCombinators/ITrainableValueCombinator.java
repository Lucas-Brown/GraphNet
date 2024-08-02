package com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators;

public interface ITrainableValueCombinator extends IValueCombinator {

    /**
     * @return The total number of variables (weights + biases)
     */
    public abstract int getNumberOfVariables();

    /**
     * Compute a unique index between 0 and {@link #getNumberOfVariables}
     * 
     * @param key the binary string corresponding to the input combination
     * @return A unique index for the specific weight
     */
    public abstract int getLinearIndexOfWeight(int key, int weight_index);

    /**
     * Compute a unique index between 0 and {@link #getNumberOfVariables}
     * 
     * @param key the binary string corresponding to the input combination
     * @return A unique index for the specific bias
     */
    public abstract int getLinearIndexOfBias(int key);

    /**
     * Get all weights and biases in linearized order.
     * Using {@link #getLinearIndexOfBias} and {@link #getLinearIndexOfWeight} to
     * access the returned array will provide the corresponding weights and biases.
     * 
     * This method is preffered to looping over all keys as some keys may share variables. 
     * @return All variables in linear order. 
     */
    public abstract double[] getLinearizedVariables();

    
    /**
     * Set all weights and biases in linearized order.
     * @see {@link #getLinearizedVariables}
     */
    public abstract void setLinearizedVariables(double[] variables);

    
    /**
     * Set all weights and biases in linearized order.
     * @see {@link #getLinearizedVariables}
     */
    public abstract void setLinearizedVariable(int index, double value);

    /**
     * Subtract the delta from each parameter.
     * The delta array is expected to correspond with the linear indices
     * 
     * @param delta The change to apply to each parameter
     * @see {@link #getLinearIndexOfBias}
     * @see {@link #getLinearIndexOfWeight}
     */
    public abstract void applyDelta(double[] delta);
}
