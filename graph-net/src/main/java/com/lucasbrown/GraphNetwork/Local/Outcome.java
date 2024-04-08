package com.lucasbrown.GraphNetwork.Local;

import com.lucasbrown.GraphNetwork.Local.Nodes.INode;

public class Outcome {
    public int binary_string;
    public double netValue;
    public double activatedValue;
    public double probability;
    public double[] sourceOutputs;
    public INode[] sourceNodes;
    public int[] sourceKeys;

    @Override 
    public String toString()
    {
        return String.format("(%.2e, %2.0f%s)", netValue, probability*100, "%");
    }

    @Override
    public boolean equals(Object o)
    {
        if(!(o instanceof Outcome)) return false;

        Outcome out = (Outcome) o;

        return binary_string == out.binary_string
            & netValue == out.netValue
            & activatedValue == out.activatedValue
            & probability == out.probability;

    }

    @Override 
    public int hashCode(){
        int hash = binary_string;
        hash *= 37;
        hash += Double.hashCode(netValue);
        hash *= 37;
        hash += Double.hashCode(activatedValue);
        hash *= 37;
        hash += Double.hashCode(probability);
        return hash;
    }

    public static int descendingProbabilitiesComparator(Outcome o1, Outcome o2){
        return (int) (o2.probability - o1.probability);
    }
}
