package com.lucasbrown.GraphNetwork.Local;

import com.lucasbrown.GraphNetwork.Local.Nodes.INode;

public class Signal{
    public final INode sendingNode;
    public final INode recievingNode;
    public final int sourceKey;
    public final double strength;
    public final double probability;

    public Signal(final INode sendingNode, final INode recievingNode, final int sourceKey, final double strength, final double probability) {
        assert Double.isFinite(strength);
        this.sendingNode = sendingNode;
        this.recievingNode = recievingNode;
        this.sourceKey = sourceKey;
        this.strength = strength;
        this.probability = probability;
    }

    public INode getSendingNode() {
        return sendingNode;
    }

    public INode getRecievingNode() {
        return recievingNode;
    }

    public int getSendingID(){
        return sendingNode.getID();
    }

    public int getRecievingID(){
        return sendingNode.getID();
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

    public static int compareSendingNodeIDs(Signal s1, Signal s2){
        return Integer.compare(s1.sendingNode.getID(), s2.sendingNode.getID());
    }
}