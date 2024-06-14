package com.lucasbrown.GraphNetwork.Global.Network;

import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;

/**
 * A collection of data which modifies the training and firing rates of each
 * node.
 * All nodes are given access to this data to use but modification should be
 * strictly controlled by {@code GraphNetwork}
 */
public class SharedNetworkData {

    public final ErrorFunction errorFunc;

    /**
     * Step size for adjusting the output values of nodes
     */
    private double epsilon;

    SharedNetworkData(ErrorFunction errorFunc, double epsilon) {
        this.errorFunc = errorFunc;
        this.epsilon = epsilon;
    }

    public double getEpsilon() {
        return epsilon;
    }

}