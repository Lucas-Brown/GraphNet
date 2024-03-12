package com.lucasbrown.NetworkTraining.GeneticAlgorithm;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.FilterDistribution;
import com.lucasbrown.GraphNetwork.Local.ICopyable;
import com.lucasbrown.GraphNetwork.Local.DataStructure.DataNode;

/**
 * An interface for training a network using a genetic algorithm
 */
public interface IGeneticTrainable extends ICopyable<IGeneticTrainable> {
    
    public abstract int getNumberOfNodes();

    public abstract int getMinimumNumberOfNodes();

    public abstract DataNode getNode(int node_id);

    /**
     * 
     * @param af
     * @return the id of the newly created ndoe
     */
    public abstract int addNewNode(ActivationFunction af);

    public abstract void removeNode(int node_id);

    public abstract void addNewConnection(int from_id, int to_id, FilterDistribution distribution);

    public abstract void removeConnection(int from_id, int to_id);

    public abstract boolean isCompatibleWith(IGeneticTrainable otherParent); 
}
