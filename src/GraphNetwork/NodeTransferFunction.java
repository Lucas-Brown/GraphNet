package src.GraphNetwork;

/**
 * Contains the probability distribution information for likelyhood of a signal being sent from one node to another
 */
public abstract class NodeTransferFunction {
    
    /**
     * The output signal strength
     */
    protected float strength;

    protected abstract boolean ShouldSend(float inputSignal);
    protected abstract float GetOutputStrength();
    protected abstract void UpdateDistribution(float backpropSignal, int N);

    /**
     * Adjusts the signal strength towards a target value 
     */
    public void AdjustSignalStrength(float target, float epsilon)
    {
        strength += (target - strength) * epsilon;
    }

    public Signal TransferSignal(Node recievingNode, float inputSignal)
    {
        // Compute whether a signal should be sent
        boolean send = ShouldSend(inputSignal);
        if(send)
        {
            // Send the signal to the next node
            Signal signal = new Signal(this, GetOutputStrength());
            recievingNode.RecieveSignal(signal);
            return signal;
        }
        return null;
    }
}
