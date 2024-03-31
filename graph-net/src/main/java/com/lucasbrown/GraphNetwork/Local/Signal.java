package com.lucasbrown.GraphNetwork.Local;

public class Signal {
    public final Node sendingNode;
    public final Node recievingNode;
    public final int sourceKey;
    public final double strength;
    public final double probability;

    public Signal(final Node sendingNode, final Node recievingNode, final int sourceKey, final double strength, final double probability) {
        assert Double.isFinite(strength);
        this.sendingNode = sendingNode;
        this.recievingNode = recievingNode;
        this.sourceKey = sourceKey;
        this.strength = strength;
        this.probability = probability;
    }

    public Node getSendingNode() {
        return sendingNode;
    }

    public Node getRecievingNode() {
        return recievingNode;
    }

    public int getSourceKey()
    {
        return sourceKey;
    }

    public double getOutputStrength() {
        return strength;
    }
    
    public double getProbability() {
        return probability;
    }
}