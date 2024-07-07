package com.lucasbrown.GraphNetwork.Local.Nodes;

/**
 * A node which exposes it's value 
 */
public class OutputNode extends NodeWrapper implements IOutputNode {

    public OutputNode(INode node) {
        super(node);
    }

    /**
     * Get the value of this node
     * The caller should first verify if this node is active using {@code isActive}
     * or get the value using {@code getValueOrNull}
     * 
     * @return
     */
    @Override
    public double getValue() {
        return 0; // TODO: fix this
    }

    /**
     * @return Checks if this node is active and returns the value if it is,
     *         otherwise returns null
     */
    @Override
    public Double getValueOrNull() {
        return hasValidForwardSignal() ? 0d : null;
    }

}
