package src.GraphNetwork.Global;

import src.GraphNetwork.Local.Node;
import src.NetworkTraining.History;

public class Signal
{
    public final Node sendingNode;
    public final Node recievingNode;
    public final double strength;
    public final History history;
    
    Signal(final Node sendingNode, final Node recievingNode, final double strength)
    {
        this(sendingNode, recievingNode, strength, null);
    }

    Signal(final Node sendingNode, final Node recievingNode, final double strength, History history)
    {
        this.sendingNode = sendingNode;
        this.recievingNode = recievingNode;
        this.strength = strength;
        this.history = history;
    }

    public Node getSendingNode()
    {
        return sendingNode;
    }

    public Node getRecievingNode()
    {
        return recievingNode;
    }

    public double getOutputStrength()
    {
        return (double) strength;
    }
}