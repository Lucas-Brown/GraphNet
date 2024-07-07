package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.NetworkTraining.History;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * Computes the gradient using a forward pass
 */
public class ForwardNetworkGradient implements INetworkGradient  {

    protected WeightsLinearizer linearizer;
    protected History<Outcome, INode> networkHistory;

    public ForwardNetworkGradient(WeightsLinearizer linearizer) {
        this.linearizer = linearizer;
    }

    @Override
    public ArrayList<HashMap<Outcome, Vec>> getGradient(History<Outcome, INode> networkHistory) {
        this.networkHistory = networkHistory;
        int n_steps = networkHistory.getNumberOfTimesteps();
        ArrayList<HashMap<Outcome, Vec>> gradientsThroughTime = new ArrayList<>(n_steps);

        for(int timestep = 0; timestep < n_steps; timestep++){
            gradientsThroughTime.add(getGradientAtTime(timestep));
        }

        return gradientsThroughTime;
    }

    private HashMap<Outcome, Vec> getGradientAtTime(int timestep) {
        HashMap<Outcome, Vec> gradientMap = new HashMap<>();
        HashMap<INode, ArrayList<Outcome>> outcomeMap = networkHistory.getStateAtTimestep(timestep);
        for (Entry<INode, ArrayList<Outcome>> entry : outcomeMap.entrySet()) {
            ITrainable node = (ITrainable) entry.getKey();
            for(Outcome outcome : entry.getValue()){
                gradientMap.put(outcome, computeGradientOfOutcome(node, outcome));
            }
        }
        return gradientMap;
    }

    protected Vec computeGradientOfOutcome(ITrainable node, Outcome outcome) {
        // the Jacobian and Hessian of the input matrix will always be zero
        if (node instanceof InputNode) {
            outcome.trainingData = new DenseVector(linearizer.totalNumOfVariables);
            return (Vec) outcome.trainingData;
        }

        Vec z_jacobi = new DenseVector(linearizer.totalNumOfVariables);
        int key = outcome.binary_string;
        double[] weights = node.getWeights(key);

        // construct the jacobian for the net value (z)
        // starting with the direct derivative of z
        for (int i = 0; i < outcome.sourceOutcomes.length; i++) {
            int idx = linearizer.getLinearIndexOfWeight(node, key, i);
            assert z_jacobi.get(idx) == 0;
            z_jacobi.set(idx, outcome.sourceOutcomes[i].activatedValue);
        }
        int bias_idx = linearizer.getLinearIndexOfBias(node, key);
        assert z_jacobi.get(bias_idx) == 0;
        z_jacobi.set(bias_idx, 1);

        // incorporate previous jacobians
        for (int i = 0; i < weights.length; i++) {
            Vec weighed_jacobi = (Vec) outcome.sourceOutcomes[i].trainingData;
            weighed_jacobi = weighed_jacobi.multiply(weights[i]);
            z_jacobi.mutableAdd(weighed_jacobi);
        }

        // apply to activated jacobi
        ActivationFunction activator = node.getActivationFunction();
        double activation_derivative = activator.derivative(outcome.netValue);
        Vec gradient = z_jacobi.multiply(activation_derivative);

        outcome.trainingData = gradient;
        return gradient;
    }
}
