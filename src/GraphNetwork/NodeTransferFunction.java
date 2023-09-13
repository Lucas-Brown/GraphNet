package src.GraphNetwork;

/**
 * Contains the probability distribution information for likelyhood of a signal being sent from one node to another
 */
public abstract class NodeTransferFunction {
    
    /**
     * The output signal strength
     */
    protected float strength;

    public abstract boolean ShouldSend(float inputSignal);
    public abstract float GetOutputStrength();
    protected abstract void UpdateDistribution(float backpropSignal, int N_Limiter);

    /**
     * Adjusts the signal strength towards a target value 
     */
    public void AdjustSignalStrength(float target, float epsilon)
    {
        strength += (target - strength) * epsilon;
    }

    public Signal GetTransferSignal(Node recievingNode, float inputSignal)
    {
        // Compute whether a signal should be sent and return the signal if it should be sent
        return ShouldSend(inputSignal) ? new Signal(this, GetOutputStrength()) : null;
    }
}
