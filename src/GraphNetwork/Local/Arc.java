package src.GraphNetwork.Local;

import java.util.Objects;

import src.GraphNetwork.Global.GraphNetwork;
import src.GraphNetwork.Global.Signal;

/**
 * A one-way connection between a sending node and a recieving node.
 * Holds the probability distribution for determining activation likelyhood  
 */
public class Arc {
    
    /**
     * The network this arc belongs to
     */
    private final GraphNetwork network;

    /**
     * Sending and recieving node
     */
    final Node sending, recieving;

    /**
     * Node transfer function for determining probability and strength of signal forwarding 
     */
    final ActivationProbabilityDistribution probDist;

    public Arc(final GraphNetwork network, final Node sending, final Node recieving, final ActivationProbabilityDistribution transferFunc)
    {
        this.network = Objects.requireNonNull(network);
        this.sending = sending;
        this.recieving = recieving;
        this.probDist = transferFunc;
    }

    public boolean doesMatchNodes(Node sendingMatch, Node recievingMatch)
    {
        return sending.equals(sendingMatch) && recieving.equals(recievingMatch);
    }

    /**
     * Attempt to send a signal from the sending node to the recieving node.
     * The probability of sending the signal is determined by the probability distribution 
     * 
     * @param strength The strength of the signal
     * @param factor A factor to multiply the probability check by
     * @return the signal or null if no signal was sent
     */
    Signal sendSignal(double strength, double factor)
    {
        // the sending node should be calling this method and should already 'know' that it is transmitting the signal
        // but for sake of clarity and to make the transferrence of a signal clear, the sending node is notified here
        if(probDist.shouldSend(strength, factor))
        {
            Signal signal = network.createSignal(sending, recieving, strength);
            sending.transmittingSignal(signal); 
            recieving.recieveSignal(signal);
        }
        return null;
    }

}
