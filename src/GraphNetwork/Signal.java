package src.GraphNetwork;

public final class Signal {
    public final NodeTransferFunction recievingFunction; 
    public final float strength;
    
    public Signal(final NodeTransferFunction recievingFunction, final float strength)
    {
        this.recievingFunction = recievingFunction;
        this.strength = strength;
    }
}
