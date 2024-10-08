package com.lucasbrown.GraphNetwork.Local;

import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.HelperClasses.WeightedAverage;

/**
 * A massive collection of everything that anyone could ever need to know about a node at a given timestep 
 */
public class Outcome {

    public INode node;
    public int binary_string;
    public double netValue;
    public double activatedValue;
    public double probability;
    public double[] sourceTransferProbabilities;
    public int[] sourceKeys;
    public Outcome[] sourceOutcomes;

    public int root_bin_str;
    public Outcome[] allRootOutcomes;

    /**
     * Accumulates the number of pass/fails for this signal.
     */
    public WeightedAverage passRate = new WeightedAverage();

    /**
     * A pointer for any data used during the training process
     */
    public Object trainingData;


    @Override
    public String toString() {
        return String.format("(%.2e, %2.0f%s, %s)", netValue, probability * 100, "%", Integer.toBinaryString(binary_string));
    }

    /*
     * @Override
     * public boolean equals(Object o)
     * {
     * if(!(o instanceof Outcome)) return false;
     * 
     * Outcome out = (Outcome) o;
     * 
     * return binary_string == out.binary_string
     * & netValue == out.netValue
     * & activatedValue == out.activatedValue
     * & probability == out.probability;
     * 
     * }
     */

    @Override
    public int hashCode() {
        int hash = binary_string;
        hash *= 37;
        hash += Double.hashCode(netValue);
        hash *= 37;
        hash += Double.hashCode(activatedValue);
        hash *= 37;
        hash += Double.hashCode(probability);
        return hash;
    }

    public static int descendingProbabilitiesComparator(Outcome o1, Outcome o2) {
        return Double.compare(o2.probability, o1.probability);
    }
}
