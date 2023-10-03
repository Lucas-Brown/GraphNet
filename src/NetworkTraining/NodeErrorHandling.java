package src.NetworkTraining;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

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
            .forEach(arc -> arc.probDist.diminishDistribution(record.nodeOutputStrength, currentNode.networkData.getDiminishmentRate()));
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
            recordList.stream().forEach(NodeErrorHandling::reinforceDistributionOfRecord);
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
            .forEach(arc -> arc.probDist.reinforceDistribution(record.nodeOutputStrength, currentNode.networkData.getReinforcmentRate()));
    }

    /**
     * Computes the error of every node steming from the root node in the history then updates the weights and biases
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
        
        // Compute the error due to the root node
        List<Record> recordList = histIter.next();
        List<Record> nextRecords1 = histIter.next();
        
        Record rootRecord = recordList.get(0); // This should only have 1 value in the list
        double first_layer_error = getErrorDerivativeOfRoot(rootRecord, target); // get the error signal of the node
        errorMap.put(rootNode, new IntegerDoublePair(rootRecord.incomingKey, first_layer_error));
        rootNode.updateWeightsAndBias(rootRecord.incomingKey, nextRecords1, first_layer_error);

        recordList = nextRecords1;
        // compute the error for all remaining nodes
        while(histIter.hasNext())
        {
            final List<Record> nextRecords = histIter.next();
            final HashMap<Node, IntegerDoublePair> finalErrorMap = errorMap; // must be final or effectively final
            HashMap<Node, IntegerDoublePair> nextErrorMap = new HashMap<>(recordList.size());

            // for each record, compute the entry of it's key and error and notify the network to update each node
            recordList.stream()
                .forEach(record -> 
                {
                    double error = getErrorDerivativeOfHidden(record, finalErrorMap);
                    record.currentNode.updateWeightsAndBias(record.incomingKey, nextRecords, error);
                    nextErrorMap.put(record.currentNode, new IntegerDoublePair(record.incomingKey, error));
                });

            errorMap = nextErrorMap;
            recordList = nextRecords;
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
        final int recordId = record.currentNode.id;
        return record.outgoingNodes.stream()
            .mapToDouble(node -> {
                IntegerDoublePair errorAndKey = errorMap.get(node);
                if(errorAndKey == null) return 0;
                return errorAndKey.doubleValue * node.getWeightOfNode(errorAndKey.intValue, recordId);
            })
            .sum();
    }
}
