package com.lucasbrown.GraphNetwork.Local;

public class Signal {
    public final Node sendingNode;
    public final Node recievingNode;
    public final double strength;

    public Signal(final Node sendingNode, final Node recievingNode, final double strength) {
        assert Double.isFinite(strength);
        this.sendingNode = sendingNode;
        this.recievingNode = recievingNode;
        this.strength = strength;
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
}