package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;

import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * Computes the gradient using a forward pass
 */
public class ForwardGradient extends ForwardMethod{
    
    private ArrayList<HashMap<

    public ForwardGradient(GraphNetwork network) {
        super(network);
    }

    
    private void computeFullErrorDerivatives() {
        for (int time = 0; time < inputs.length; time++) {
            for (ITrainable node : allNodes) {
                computeFullErrorDerivatives(node, time);
            }
        }
    }

    protected void computeFullErrorDerivatives(ITrainable node, int timestep) {
        ArrayList<Outcome> outcomes = networkHistory.getStateOfNode(timestep, node);

        if (outcomes == null || outcomes.isEmpty()) {
            return;
        }

        // initialize matrices
        for (Outcome outcome : outcomes) {
            outcome.errorJacobian = new DenseVector(totalNumOfVariables);
            outcome.errorHessian = new DenseMatrix(totalNumOfVariables, totalNumOfVariables);
        }

        // Compute the Jacobians and Hessians
        for (Outcome outcome : outcomes) {
            computeJacobian(node, outcome);
        }
    }

    protected Vec computeJacobian(ITrainable node, Outcome outcome) {
        // the Jacobian and Hessian of the input matrix will always be zero
        if (node instanceof InputNode) {
            return null;
        }

        Vec z_jacobi = new DenseVector(totalNumOfVariables);
        int key = outcome.binary_string;
        double[] weights = node.getWeights(key);

        // construct the jacobian for the net value (z)
        // starting with the direct derivative of z
        for (int i = 0; i < outcome.sourceOutcomes.length; i++) {
            int idx = getLinearIndexOfWeight(node, key, i);
            z_jacobi.set(idx, outcome.sourceOutcomes[i].activatedValue);
        }
        int bias_idx = getLinearIndexOfBias(node, key);
        z_jacobi.set(bias_idx, 1);

        // incorporate previous jacobians
        for (int i = 0; i < weights.length; i++) {
            Outcome so = outcome.sourceOutcomes[i];
            Vec weighed_jacobi = so.errorJacobian.multiply(weights[i]);
            z_jacobi.mutableAdd(weighed_jacobi);
        }

        // apply to activated jacobi
        ActivationFunction activator = node.getActivationFunction();
        double activation_derivative = activator.derivative(outcome.netValue);
        outcome.errorJacobian = z_jacobi.multiply(activation_derivative);

        return z_jacobi;

    }

    @Override
    public Vec getGradient() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getGradient'");
    }

}
