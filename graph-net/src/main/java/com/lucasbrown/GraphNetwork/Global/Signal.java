package com.lucasbrown.GraphNetwork.Global;

import com.lucasbrown.GraphNetwork.Local.Node;

public class Signal {
    public final Node sendingNode;
    public final Node recievingNode;
    public final double strength;

    Signal(final Node sendingNode, final Node recievingNode, final double strength) {
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