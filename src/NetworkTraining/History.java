package src.NetworkTraining;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;

/**
 * Contains the entire history of a signal being passed through the network
 * Since signals may split off and re-join and merge with other signals, A history may become highly non-linear.
 * Histories are primarily used for training and debugging.
 */
public class History {
    
    /**
     * Every node that the signal(s) have interacted with  
     */
    private final LinkedList<ArrayList<Record>> entireHistory;

    /**
     * History is in the making! 
     * Stores all records generated this time step.  
     */
    private ArrayList<Record> currentRecords;

    public History()
    {
        entireHistory = new LinkedList<>();
        currentRecords = new ArrayList<>();
        entireHistory.add(currentRecords);
    }

    /**
     * Add the record to the list of all records this time step
     * @param node
     */
    public void addToCurrentRecord(Record node)
    {
        // the current step should never contain more than one of the same node
        if(currentRecords.contains(node))
        {
            throw new RuntimeException("Current records already contain this record: \n" + node.toString());
        }
        else
        {
            currentRecords.add(node);
        }
    }

    /**
     * History steps forward and starts recording the new era
     */
    public void step()
    {
        // The size of {@code currentStep} should only change in the event of a history merge
        currentRecords.trimToSize(); 
        currentRecords = new ArrayList<>();
        entireHistory.add(currentRecords);
    }

    public History mergeHistories(History h1, History h2)
    {
        History mergedHistory = new History();

        // Ensure h1 is the larger of the histories
        // This is necessary for later when merging back through time 
        if(h1.entireHistory.size() < h2.entireHistory.size())
        {
            // swap h1 and h2
            History temp = h1;
            h1 = h2;
            h2 = temp;
        }

        // Find all history nodes that overlap
        HashSet<Record> intersection = new HashSet<>(h1.currentRecords);
        intersection.retainAll(h2.currentRecords);

        if(intersection.size() == 0)
        {
            System.err.println("Warning: two histories were merged without any current step overlap. This can cause history records to grow indefinitely and impact training performance.");
        }

        // merge the current step
        mergedHistory.currentRecords.addAll(new ArrayList<>(h1.currentRecords));
        mergedHistory.currentRecords.removeAll(intersection); // this order matters!!
        mergedHistory.currentRecords.addAll(new ArrayList<>(h2.currentRecords));

        
        // Begin merging the histories backwards.
        // No two history steps (other than the current step) should have any overlap
        Iterator<ArrayList<Record>> h1Iter = h1.entireHistory.descendingIterator();
        Iterator<ArrayList<Record>> h2Iter = h2.entireHistory.descendingIterator();
        h1Iter.next(); // skip the current step 
        h2Iter.next();

        while(h2Iter.hasNext())
        {
            ArrayList<Record> mergedStep = new ArrayList<>(h1Iter.next());
            mergedStep.addAll(h2Iter.next());
            mergedHistory.entireHistory.addFirst(mergedStep);
            assert mergedStep.size() == (new HashSet<>(mergedStep)).size(): "Failed to merge step; History has been corrupted! \nboth histories contain the same record. This is invalid and should have resulted in a merge event.";
        }

        // Creating a new arrayList isn't neccessary but is good practice to avoid issues.
        h1Iter.forEachRemaining(hNode -> mergedHistory.entireHistory.addFirst(new ArrayList<>(hNode)));

        return mergedHistory;
    }
}
