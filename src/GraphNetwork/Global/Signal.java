package src.GraphNetwork.Global;

import src.GraphNetwork.Local.Node;

public class Signal
{
    public final Node sendingNode;
    public final float strength;
    
    Signal(final Node sendingNode, final Node recievingNode, final float strength)
    {
        this.sendingNode = sendingNode;
        this.strength = strength;
    }

    public double getOutputStrength()
    {
        return (double) strength;
    }
}