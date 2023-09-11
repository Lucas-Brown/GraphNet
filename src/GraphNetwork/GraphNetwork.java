package src.GraphNetwork;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * A neural network using a probabalistic directed graph representation.
 * Training currently a work in progress
 */
public class GraphNetwork {

    /**
     * Estimator for the number of data points each distribution represents
     */
    private Integer N_estimator; 

    /**
     * Step size for adjusting the output values of nodes
     */
    private float epsilon;


    /**
     * A list of all nodes within the graph network
     */
    public ArrayList<Node> nodes;

    public Consumer<HashSet<Node>> corrector;


    /**
     * A has set containing every node that recieved a signal in the last step
     */
    private HashSet<Node> signaledNodes;

    public GraphNetwork()
    {
        N_estimator = 100;
        epsilon = 0.001f;

        nodes = new ArrayList<>();
        signaledNodes = new HashSet<>();
    }

    public void Step()
    {
        HashSet<Node> nextSignaledNodes = new HashSet<>();

        // Step every node forward
        signaledNodes.forEach(Node::HandleIncomingSignals);

        // Correct node misfires and firing values
        if(corrector != null)
        {
            corrector.accept(signaledNodes);
        }

        // Send next signals
        nextSignaledNodes = signaledNodes.stream().flatMap(Node::TransmitSignal).collect(Collectors.toCollection(HashSet::new));

        // Adjust transfer distributions
        signaledNodes.forEach(n -> n.CorrectRecievingValue(epsilon));
        signaledNodes.forEach(n -> n.ReinforceSignalPathways(N_estimator));
        signaledNodes = nextSignaledNodes;
    }
}
