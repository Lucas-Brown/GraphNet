package src.NetworkTraining;

import java.util.AbstractMap.SimpleEntry;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import src.GraphNetwork.Global.GraphNetwork;
import src.GraphNetwork.Local.Node;

/**
 * A collection of static methods which are used for training nodes 
 */
public class NodeErrorHandling {
    
    /**
     * Diminish the firing chance of all distributions in the history that reach the root node
     * @param history
     * @param rootNode
     */
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
            .forEach(arc -> arc.probDist.diminishDistribution(record.nodeOutputStrength));
    }

    /**
     * Reinforce the firing chance of all distributions in the history that reach the root node
     * @param history
     * @param rootNode
     */
    public static void reinforceFiringChances(History history, Node rootNode)
    {
        history.getNodeHistoryIterator(rootNode).forEachRemaining(recordList -> 
        {
            recordList.stream().forEach(NodeErrorHandling::diminishDistributionOfRecord);
        });
    }

    public static void reinforceDistributionOfRecord(Record record)
    {
        Node currentNode = record.currentNode;
        /*
         * Find the arc associated with the transfer between the current node and the output node
         * Then, diminish the probability of that node  
         */
        record.getOutgoingNodes().stream()
            .map(currentNode::getArc)
            .forEach(arc -> arc.probDist.reinforceDistribution(record.nodeOutputStrength, currentNode.networkData.getN_Limiter()));
    }

    /**
     * Compute the error of every node steming from the root node in the history.
     * All errors are assigned to the node and the network as alerted that each node should be queued to have its weights/biases updated.
     * @param history The entire history of the signal leading up to the root node
     * @param rootNode The root node which the signal reached
     * @param target The target value for this node
     */
    public static void computeErrorSignalsOfHistory(History history, Node rootNode, double target)
    {
        // map nodes to errors from THIS operation
        // cannot use rootNode.errorSignal since it is an accumulative arror 
        HashMap<Node, IntegerDoublePair> errorMap = new HashMap<>(); 
        Iterator<List<Record>> histIter = history.getNodeHistoryIterator(rootNode);
        GraphNetwork gn = rootNode.network;
        
        // Compute the error due to the root node
        Record rootRecord = histIter.next().get(0); // This should only have 1 value in the list
        int bitStr = rootNode.nodeSetToBinStr(rootRecord.incomingNodes);
        double error = getErrorDerivativeOfRoot(rootRecord, target); // get the error signal of the node
        errorMap.put(rootNode, new IntegerDoublePair(bitStr, error));
        rootNode.addToError(bitStr, error);
        gn.notifyErrorUpdateRequired(rootNode);

        // compute the error for all remaining nodes
        while(histIter.hasNext())
        {
            List<Record> recordList = histIter.next();
            final HashMap<Node, IntegerDoublePair> finalErrorMap = errorMap; // must be final or effectively final
            HashMap<Node, IntegerDoublePair> nextErrorMap = new HashMap<>(recordList.size());

            // for each record, compute the entry of it's key and error and notify the network to update each node
            recordList.stream()
                .map(record -> getEntryOfNode(record, finalErrorMap))
                .forEach(entry -> 
                {
                    Node node = entry.getKey();
                    gn.notifyErrorUpdateRequired(node); 
                    nextErrorMap.put(node, entry.getValue());
                });

            errorMap = nextErrorMap;
        }
    }

    public static void sendErrorSignal()
    {

    }

    /**
     * Get the derivative of the error for the root node
     * @param rootRecord The record of the root node
     * @param target the target value for the node
     * @return the error signal to correct the output of the root node towards the target
     */
    private static double getErrorDerivativeOfRoot(Record rootRecord, double target)
    {
        double d_mse = rootRecord.currentNode.networkData.errorFunc.error_derivative(rootRecord.nodeOutputStrength, target);
        double d_activation = rootRecord.currentNode.activationFunction.derivative(rootRecord.nodeSignalStrength);
        return d_mse * d_activation;
    }

    private static SimpleEntry<Node, IntegerDoublePair> getEntryOfNode(Record record, HashMap<Node, IntegerDoublePair> errorMap)
    {
        int bitStr = record.currentNode.nodeSetToBinStr(record.incomingNodes);
        double error = getErrorDerivativeOfHidden(record, errorMap);
        return new SimpleEntry<Node, IntegerDoublePair>(record.currentNode, new IntegerDoublePair(bitStr, error));
    }

    /**
     * Get the derivative of the error for the root node
     * @param record The record of the root node
     * @param errorMap the map of node to error 
     * @return the error signal to correct the output of the root node towards the target
     */
    private static double getErrorDerivativeOfHidden(Record record, HashMap<Node, IntegerDoublePair> errorMap)
    {
        double error = getAccumulatedErrorOfHiddenNode(record, errorMap);
        double d_activation = record.currentNode.activationFunction.derivative(record.nodeSignalStrength);
        return error * d_activation;
    }

    private static double getAccumulatedErrorOfHiddenNode(Record record, HashMap<Node, IntegerDoublePair> errorMap)
    {
        return record.outgoingNodes.stream()
            .mapToDouble(node -> {
                IntegerDoublePair pair = errorMap.get(node);
                return pair.doubleValue * node.getWeightOfNode(pair.intValue, node.id);
            })
            .sum();
    }
}
