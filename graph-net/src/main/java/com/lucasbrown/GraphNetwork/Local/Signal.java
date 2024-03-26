package com.lucasbrown.GraphNetwork.Local;

public class Signal {
    public final Node sendingNode;
    public final Node recievingNode;
    public final double strength;
    public final double probability;

    public Signal(final Node sendingNode, final Node recievingNode, final double strength, final double probability) {
        assert Double.isFinite(strength);
        this.sendingNode = sendingNode;
        this.recievingNode = recievingNode;
        this.strength = strength;
        this.probability = probability;
    }

    public Node getSendingNode() {
        return sendingNode;
    }

    public Node getRecievingNode() {
        return recievingNode;
    }

    public double getOutputStrength() {
        return strength;
    }
    
    public double getProbability() {
        return probability;
    }
}