package com.lucasbrown.GraphNetwork.Local;

import com.lucasbrown.GraphNetwork.Local.Nodes.INode;

public class Signal{
    public final INode sendingNode;
    public final INode recievingNode;
    public final Outcome sourceOutcome;
    public final double firingProbability;

    public Signal(final INode sendingNode, final INode recievingNode, final Outcome sourceOutcome, final double firingProbability) {
        assert Double.isFinite(sourceOutcome.activatedValue);
        this.sendingNode = sendingNode;
        this.recievingNode = recievingNode;
        this.sourceOutcome = sourceOutcome;
        this.firingProbability = firingProbability;
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
        return sourceOutcome.binary_string;
    }

    public double getOutputStrength() {
        return sourceOutcome.activatedValue;
    }
    
    public double getProbability() {
        return sourceOutcome.probability * firingProbability;
    }

    public static int compareSendingNodeIDs(Signal s1, Signal s2){
        return Integer.compare(s1.sendingNode.getID(), s2.sendingNode.getID());
    }
}