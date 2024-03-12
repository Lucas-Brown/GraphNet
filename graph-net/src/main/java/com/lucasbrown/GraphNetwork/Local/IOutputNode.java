package com.lucasbrown.GraphNetwork.Local;

public interface IOutputNode {
    
    /**
     * Get the value of this node
     * The caller should first verify if this node is active using {@code isActive}
     * or get the value using {@code getValueOrNull}
     * 
     * @return
     */
    public abstract double getValue();

    /**
     * @return Checks if this node is active and returns the value if it is,
     *         otherwise returns null
     */
    public abstract Double getValueOrNull();

    public abstract void acceptUserBackwardSignal(double value);
}
