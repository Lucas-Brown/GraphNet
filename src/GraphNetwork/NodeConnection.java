package src.GraphNetwork;

/**
 * A one-way connection between a sending node and a recieving node.
 * Holds the sending probability distribution  
 */
public class NodeConnection {
    
    /**
     * Sending and recieving node
     */
    Node sending, recieving;

    /**
     * Node transfer function for determining probability and strength of signal forwarding 
     */
    NodeTransferFunction transferFunc;

    public NodeConnection(Node sending, Node recieving, NodeTransferFunction transferFunc)
    {
        this.sending = sending;
        this.recieving = recieving;
        this.transferFunc = transferFunc;
    }

    public boolean DoesMatchNodes(Node sendingMatch, Node recievingMatch)
    {
        return sending.equals(sendingMatch) && recieving.equals(recievingMatch);
    }

    /**
     * Send a signal from the sending node to the recieving node
     * @param strength The strength of the signal
     * @return the signal or null if no signal was sent
     */
    public Signal SendSignal(float strength)
    {
        return transferFunc.TransferSignal(recieving, strength);
    }

}
