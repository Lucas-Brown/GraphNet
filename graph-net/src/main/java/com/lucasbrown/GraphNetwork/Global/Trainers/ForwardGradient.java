package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.NetworkTraining.History;
import com.lucasbrown.NetworkTraining.IStateGenerator;
import com.lucasbrown.NetworkTraining.IStateRecord;

import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * Computes the gradient using a forward pass
 */
public class ForwardGradient extends ForwardMethod {

    private History<Outcome, INode> networkHistory;

    private ArrayList<HashMap<Outcome, Vec>> gradientsThroughTime;

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

    @Override
    public ArrayList<HashMap<Outcome, Vec>> getGradient(History<Outcome, INode> networkHistory) {
        this.networkHistory = networkHistory;
        int n_steps = networkHistory.getNumberOfTimesteps();
        gradientsThroughTime = new ArrayList<>(n_steps);

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
                gradientMap.put(outcome, getGradientOfOutcome(node, outcome, timestep));
            }
        }
        return gradientMap;
    }

    private Vec getGradientOfOutcome(ITrainable node, Outcome outcome, int timestep) {
        
    }

    
    public Vec[] getPreviousGradients(ITrainable node, Outcome outcome, int timestep) {
        Vec[] prev_gradients = new Vec[outcome.sourceNodes.length];
        if (prev_gradients.length == 0) {
            return prev_gradients;
        }

        for (int i = 0; i < outcome.sourceOutcomes.length; i++) {
            Outcome so = outcome.sourceOutcomes[i];
            ITrainable sourceNode = (ITrainable) outcome.sourceNodes[i];
            // get all possible gradients
            ArrayList<Gradient> gradientsToCheck = gradientHistory.getStateOfRecord(timestep - 1,
                    new GradientGenerator(sourceNode));

            // find the gradient corresponding to this outcome
            for (Vec grad : gradientsToCheck) {
                if (grad.outcome == so) {
                    prev_gradients[i] = grad;
                }
            }
            assert Objects.nonNull(prev_gradients[i]);
        }
    }

    protected Vec computeJacobian() {
        // the Jacobian and Hessian of the input matrix will always be zero
        if (node instanceof InputNode) {
            return null;
        }

        gradient = new DenseVector(totalNumOfVariables);
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
            Vec weighed_jacobi = prev_gradients[i].gradient.multiply(weights[i]);
            z_jacobi.mutableAdd(weighed_jacobi);
        }

        // apply to activated jacobi
        ActivationFunction activator = node.getActivationFunction();
        double activation_derivative = activator.derivative(outcome.netValue);
        gradient = z_jacobi.multiply(activation_derivative);

        return z_jacobi;

    }
}
