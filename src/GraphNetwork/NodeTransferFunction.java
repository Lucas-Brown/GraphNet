package src.GraphNetwork;

/**
 * Contains the probability distribution information for likelyhood of a signal being sent from one node to another
 */
public interface NodeTransferFunction {
    
    public abstract boolean ShouldSend(float inputSignal);
    public abstract float GetOutputSignal();
}
