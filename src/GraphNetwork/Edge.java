package src.GraphNetwork;

/**
 * A one-way connection between a sending node and a recieving node.
 * Holds the sending probability distribution  
 */
public class Edge {
    
    /**
     * Sending and recieving node
     */
    Node sending, recieving;

    /**
     * Node transfer function for determining probability and strength of signal forwarding 
     */
    NodeTransferFunction transferFunc;

    public Edge(Node sending, Node recieving, NodeTransferFunction transferFunc)
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
     * @param factor A factor to multiply the probability check by
     * @return the signal or null if no signal was sent
     */
    public Signal SendSignal(float strength, float factor)
    {
        // the sending node should be calling this method and should already 'know' that it is transmitting the signal
        // but for sake of clarity and to make the transferrence of a signal clear, the sending node is notified here
        Signal signal = transferFunc.GetTransferSignal(recieving, strength, factor);
        if(signal != null)
        {
            sending.NotifyTransmittingSignal(signal); 
            recieving.NotifyRecieveSignal(signal);
        }
        return signal;
    }

}
