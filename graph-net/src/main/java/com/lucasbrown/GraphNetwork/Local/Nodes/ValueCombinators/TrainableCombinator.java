package com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators;

public abstract class TrainableCombinator extends SignalCombinator{
    
    /**
     * Compute a unique index between 0 and {@link SignalCombinator#getNumberOfVariables} 
     * @param key the binary string corresponding to the input combination
     * @return A unique index for the specific weight
     */
    public abstract int getLinearIndexOfWeight(int key, int weight_index);

    /**
     * Compute a unique index between 0 and {@link SignalCombinator#getNumberOfVariables} 
     * @param key the binary string corresponding to the input combination
     * @return A unique index for the specific bias
     */
    public abstract int getLinearIndexOfBias(int key);

    /**
     * Subtract the delta from each parameter. 
     * The delta array is expected to correspond with the linear indices
     * @param delta The change to apply to each parameter
     * @see {@link #getLinearIndexOfBias} 
     * @see {@link #getLinearIndexOfWeight}
     */
    public abstract void applyDelta(double[] delta);
}
