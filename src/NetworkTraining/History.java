package src.NetworkTraining;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;

import src.GraphNetwork.Local.Node;

/**
 * Contains the entire history of a signal being passed through the network
 * Since signals may split off and re-join and merge with other signals, A history may become highly non-linear.
 * Histories are primarily used for training and debugging.
 */
public class History {
    
    /**
     * Every node that the signal(s) have interacted with  
     */
    private final LinkedList<HashMap<Node, Record>> entireHistory;

    /**
     * History is in the making! 
     * Stores all records generated this time step.  
     */
    private HashMap<Node, Record> currentRecords;

    public History()
    {
        entireHistory = new LinkedList<>();
        currentRecords = new HashMap<>();
        entireHistory.add(currentRecords);
    }

    /**
     * Add the record to the list of all records this time step
     * @param record
     */
    public void addToCurrentRecord(Record record)
    {
        // the current step should never contain more than one of the same node
        if(currentRecords.containsKey(record))
        {
            throw new RuntimeException("Current records already contain this record: \n" + record.toString());
        }
        else
        {
            currentRecords.put(record.currentNode, record);
        }
    }

    /**
     * History steps forward and starts recording the new era
     */
    public void step()
    {
        currentRecords.keySet().stream().forEach(node -> node.setHistory(null));
        currentRecords = new HashMap<>();
        entireHistory.add(currentRecords);
    }

    /**
     * Starting from the record which contains the {@code staringNode}, delete all connected records containing a single outgoing signal 
     * In other words, delete the chain of signals back to the last node which branched
     * @param startingNode 
     */
    public void decimateTimeline(Node startingNode)
    {
        // get the record associated with the node
        Iterator<HashMap<Node, Record>> histIter = entireHistory.descendingIterator();
        List<Node> nodesToDelete = new ArrayList<>(1);
        List<Record> recordsToDelete = new ArrayList<>(1);
        nodesToDelete.add(startingNode);
        recordsToDelete.add(currentRecords.get(startingNode));

        // remove the values at the current step
        HashMap<Node, Record> recordsAtStep = histIter.next();
        recordsAtStep.values().removeAll(recordsToDelete);

        boolean stillRemovingRecords = true;
        while(stillRemovingRecords && histIter.hasNext())
        {
            recordsAtStep = histIter.next();
            List<Node> lastNodes = nodesToDelete;
            
            // get the next set of nodes
            nodesToDelete = recordsToDelete.stream().flatMap(Record::getIncomingNodesStream).toList(); 

            recordsToDelete = nodesToDelete.stream()
                .map(recordsAtStep::get) // map each node to it's corresponding record
                .map(record -> {record.outgoingNodes.removeAll(lastNodes); return record;}) // remove references to previous set
                .filter(Record::hasNoOutputSignal) // only remove records with no output signals
                .toList();

            // update nodes to only be those that have singular outputs
            nodesToDelete = recordsToDelete.stream().map(Record::getCurrentNode).toList();
                
            // delete all the records
            recordsAtStep.values().removeAll(recordsToDelete);
        }
    }

    public Iterator<List<Record>> getNodeHistoryIterator(Node rootNode)
    {
        return new NodeHistoryIterator(rootNode);
    }
    
    public static History mergeHistories(HashSet<History> histories)
    {
        History mergedHistory = new History();

        // merge the current step
        /* 
        mergedHistory.currentRecords = histories.stream()
            .flatMap(hist -> hist.currentRecords.stream())
            .distinct()
            .collect(Collectors.toCollection(ArrayList::new));
            */

        
        // Begin merging the histories backwards.
        // No two history steps (other than the current step) should have any overlap
        List<Iterator<HashMap<Node, Record>>> histIters = histories.stream()
            .map(hist -> hist.entireHistory.descendingIterator())
            .toList();
        
        // skip the current step 
       // histIters.stream().forEach(Iterator::next);

        while(!histIters.isEmpty())
        {
            // collect all record arrays into one massive record array
            HashMap<Node, Record> mergedStep = new HashMap<Node, Record>();
            histIters.stream()
                .map(iter -> iter.next())
                .forEach(subMap -> mergedStep.putAll(subMap));

            // remove any iterators that have no more elements
            histIters = histIters.stream().filter(Iterator::hasNext).toList();
                
            mergedHistory.entireHistory.addFirst(mergedStep);
            // assertion would need to be reworked 
            //assert mergedStep.size() == (new HashSet<>(mergedStep)).size(): "Failed to merge step; History has been corrupted! \nboth histories contain the same record. This is invalid and should have resulted in a merge event.";
        }

        return mergedHistory;
    }


    private class NodeHistoryIterator implements Iterator<List<Record>> {

        private List<Node> currentStep;
        private final Iterator<HashMap<Node, Record>> entireHistoryIterator;

        public NodeHistoryIterator(Node rootNode)
        {
            currentStep = new ArrayList<>(1);
            currentStep.add(rootNode);
            entireHistoryIterator = entireHistory.descendingIterator();
        }

        @Override
        public boolean hasNext() {
            return entireHistoryIterator.hasNext();
        }

        @Override
        public List<Record> next() 
        {
            HashMap<Node, Record> map = entireHistoryIterator.next(); // get the mapping from nodes to records
            List<Record> records = currentStep.stream().map(map::get).toList(); // map nodes to records
            assert !records.contains(null);
            currentStep = records.stream()
                .flatMap(Record::getIncomingNodesStream)
                .distinct()
                .toList(); // get every node which transferred a signal to this node for the next iteration
            return records;
        }
    
        
    }
    
}
