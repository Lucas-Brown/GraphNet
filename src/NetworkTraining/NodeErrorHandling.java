package src.NetworkTraining;

import src.GraphNetwork.Local.Node;

/**
 * A collection of static methods which are used for training nodes 
 */
public class NodeErrorHandling {
    
    public static void diminishFiringChances(History history, Node rootNode)
    {
        history.getNodeHistoryIterator(rootNode).forEachRemaining(recordList -> 
        {
            recordList.stream().forEach(NodeErrorHandling::diminishDistributionOfRecord);
        });
    }

    public static void diminishDistributionOfRecord(Record record)
    {
        Node currentNode = record.currentNode;
        /*
         * Find the arc associated with the transfer between the current node and the output node
         * Then, diminish the probability of that node  
         */
        record.getOutgoingNodes().stream()
            .map(currentNode::getArc)
            .forEach(arc -> arc.probDist.diminishDistribution(record.nodeSignalStrength));
    }

    public static void correctSignalValue(double target)
    {
        if(history == null) return; // can't correct signal value without a full history back to an input node
        double mse = NetworkError.MSE(mergedSignalStrength, target);

    }

    public static void sendErrorSignal()
    {

    }
}
