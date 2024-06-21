package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.IInputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.History;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.WeightedAverage;
import com.lucasbrown.NetworkTraining.DataSetTraining.IExpectationAdjuster;

import jsat.linear.DenseMatrix;
import jsat.linear.LUPDecomposition;
import jsat.linear.Matrix;
import jsat.linear.Vec;

public class NewtonTrainer {

    public double epsilon = 0.01;
    private WeightedAverage total_error;

    private int timestep;
    private final GraphNetwork network;
    private final History networkHistory;
    private final ErrorFunction errorFunction;

    private Double[][] inputs;
    private Double[][] targets;

    private ArrayList<OutputNode> outputNodes;
    private HashSet<ITrainable> allNodes;

    private int totalNumOfVariables;
    private HashMap<ITrainable, Integer> vectorNodeOffset;

    private Matrix parameterDeltas;
    private Matrix errorDerivative;
    private Matrix errorHessian;

    // consider a mask instead?
    private boolean[] isColumnEmpty;

    public NewtonTrainer(GraphNetwork network, ErrorFunction errorFunction) {
        this.network = network;
        this.errorFunction = errorFunction;
        networkHistory = new History(network);

        castAllToTrainable();

        network.setInputOperation(this::applyInputToNode);
        outputNodes = network.getOutputNodes();

        total_error = new WeightedAverage();
    }

    private void castAllToTrainable() {
        ArrayList<INode> nodes = network.getNodes();
        vectorNodeOffset = new HashMap<>(nodes.size());
        allNodes = new HashSet<>(nodes.size());

        totalNumOfVariables = 0;
        for (INode node : nodes) {
            ITrainable tnode = (ITrainable) node;
            allNodes.add(tnode);
            vectorNodeOffset.put(tnode, totalNumOfVariables);
            totalNumOfVariables += tnode.getNumberOfVariables();
        }
    }

    /**
     * input and target dimension : [timestep][node]
     * 
     * @param inputs
     * @param targets
     */
    public void setTrainingData(Double[][] inputs, Double[][] targets) {
        this.inputs = inputs;
        this.targets = targets;
    }

    public void trainNetwork(int steps, int print_interval) {
        while (steps-- > 0) {
            trainingStep(steps % print_interval == 0);
        }
    }

    public void trainingStep(boolean print_forward) {
        captureForward(print_forward);

        computeFullErrorDerivatives();
        computeDelta(print_forward);
        backpropagateErrors();
        applyErrorSignals();
        network.deactivateAll();
        networkHistory.burnHistory();
    }

    private int getLinearIndexOfWeight(ITrainable node, int key, int weight_index) {
        return vectorNodeOffset.get(node) + node.getLinearIndexOfWeight(key, weight_index);
    }

    private int getLinearIndexOfBias(ITrainable node, int key) {
        return vectorNodeOffset.get(node) + node.getLinearIndexOfBias(key);
    }

    private void captureForward(boolean print_forward) {
        for (timestep = 0; timestep < inputs.length; timestep++) {
            network.trainingStep();
            if (print_forward) {
                System.out.println(network.toString() + " | Target = " + Arrays.toString(targets[timestep]));
            }
            networkHistory.captureState();
        }
        timestep--;
    }

    private void computeFullErrorDerivatives() {
        for (int time = 0; time < inputs.length; time++) {
            for (ITrainable node : allNodes) {
                computeFullErrorDerivatives(node, time);
            }
        }
    }

    private void computeFullErrorDerivatives(ITrainable node, int timestep) {
        ArrayList<Outcome> outcomes = networkHistory.getStateOfNode(timestep, node);

        if (outcomes == null || outcomes.isEmpty()) {
            return;
        }

        // initialize matrices
        for (Outcome outcome : outcomes) {
            outcome.errorJacobian = new DenseMatrix(totalNumOfVariables, 1);
            outcome.errorHessian = new DenseMatrix(totalNumOfVariables, totalNumOfVariables);
        }

        // Compute the Jacobians and Hessians
        for (Outcome outcome : outcomes) {
            computeErrorSignals(node, outcome);
        }
    }

    /**
     * 
     * @param node
     * @param outcome
     * @param probabilityVolume
     */
    private void computeErrorSignals(ITrainable node, Outcome outcome) {
        // the Jacobian and Hessian of the input matrix will always be zero
        if (node instanceof InputNode) {
            return;
        }

        Matrix z_jacobi = new DenseMatrix(totalNumOfVariables, 1);
        int key = outcome.binary_string;
        double[] weights = node.getWeights(key);

        // construct the jacobian for the net value (z)
        // starting with the direct derivative of z
        for (int i = 0; i < outcome.sourceOutcomes.length; i++) {
            int idx = getLinearIndexOfWeight(node, key, i);
            z_jacobi.set(idx, 0, outcome.sourceOutcomes[i].activatedValue);
        }
        int bias_idx = getLinearIndexOfBias(node, key);
        z_jacobi.set(bias_idx, 0, 1);

        // incorporate previous jacobians
        for (int i = 0; i < weights.length; i++) {
            Outcome so = outcome.sourceOutcomes[i];
            Matrix weighed_jacobi = so.errorJacobian.multiply(weights[i]);
            z_jacobi.mutableAdd(weighed_jacobi);
        }

        // use the net value jacobian to compute the activation jacobian
        ActivationFunction activator = node.getActivationFunction();
        double activation_derivative = activator.derivative(outcome.netValue);
        double activation_second_derivative = activator.secondDerivative(outcome.netValue);

        // construct hessian
        Matrix JJT = z_jacobi.multiplyTranspose(z_jacobi);
        Matrix jacobi_chain = new DenseMatrix(totalNumOfVariables, totalNumOfVariables);

        for (int i = 0; i < outcome.sourceOutcomes.length; i++) {
            int idx = getLinearIndexOfWeight(node, key, i);
            Outcome so = outcome.sourceOutcomes[i];
            Matrix jac = so.errorJacobian;
            for (int j = 0; j < totalNumOfVariables; j++) {
                jacobi_chain.set(j, idx, jac.get(j, 0));
            }
        }

        jacobi_chain.mutableAdd(jacobi_chain.transpose());

        for (int i = 0; i < outcome.sourceOutcomes.length; i++) {
            Outcome so = outcome.sourceOutcomes[i];
            jacobi_chain.mutableAdd(so.errorHessian.multiply(weights[i]));
        }

        // finalize Hessian
        outcome.errorHessian = JJT.multiply(activation_second_derivative);
        outcome.errorHessian.mutableAdd(jacobi_chain.multiply(activation_derivative));
        
        // apply to activated jacobi
        outcome.errorJacobian = z_jacobi.multiply(activation_derivative);

    }

    private void computeDelta(boolean print_forward) {
        errorDerivative = new DenseMatrix(totalNumOfVariables, 1);
        errorHessian = new DenseMatrix(totalNumOfVariables, totalNumOfVariables);

        computeErrorOfOutput(print_forward);

        // double check hesssian symmetry
        for (int i = 0; i < totalNumOfVariables; i++) {
            for (int j = i + 1; j < totalNumOfVariables; j++) {
                assert errorHessian.get(i, j) == errorHessian.get(j, i);
            }
        }

        Matrix errorDerivativeNonZero = removeZeroColumns();

        LUPDecomposition decomposition = new LUPDecomposition(errorHessian);

        // if the matrix can't be practically inverted, use a simple gradient step
        // if(Math.abs(decomposition.det()) < 1E-6){
        // parameterDeltas = errorDerivative.multiply(epsilon/inputs.length);
        // return;
        // }

        Matrix non_zero_deltas = decomposition.solve(errorDerivativeNonZero);

        // ensure that we aim for the minimum instead of the maximum
        for (int i = 0; i < non_zero_deltas.rows(); i++) {
            non_zero_deltas.set(i, 0,
                    Math.abs(non_zero_deltas.get(i, 0)) * Math.signum(errorDerivativeNonZero.get(i, 0)));
        }

        // recombine deltas
        parameterDeltas = new DenseMatrix(totalNumOfVariables, 1);
        int non_empty_idx = 0;
        for (int i = 0; i < totalNumOfVariables; i++) {
            double value;
            if (isColumnEmpty[i]) {
                value = errorDerivative.get(i, 0);
            } else {
                value = non_zero_deltas.get(non_empty_idx++, 0);
            }

            parameterDeltas.set(i, 0, value);
        }

        // for (double d : parameterDeltas.getColumn(0).arrayCopy()) {
        // assert Double.isFinite(d);
        // }
        for (int i = 0; i < totalNumOfVariables; i++) {
            if (Double.isNaN(parameterDeltas.get(i, 0))) {
                parameterDeltas.set(i, 0, 0.00001*errorDerivative.get(i, 0)/inputs.length);
            }
        }
    }

    private Matrix removeZeroColumns() {

        isColumnEmpty = new boolean[totalNumOfVariables];
        int n_nnz = 0;
        for (int i = 0; i < totalNumOfVariables; i++) {
            isColumnEmpty[i] = errorHessian.getColumn(i).nnz() == 0;
            if (!isColumnEmpty[i]) {
                n_nnz++;
            }
        }

        Matrix new_derivative = new DenseMatrix(n_nnz, 1);
        Matrix new_hessian = new DenseMatrix(n_nnz, n_nnz);

        int non_zero_index = 0;
        for (int i = 0; i < totalNumOfVariables; i++) {
            if (!isColumnEmpty[i]) {
                new_derivative.set(non_zero_index, 0, errorDerivative.get(i, 0));

                Vec new_vec = new_hessian.getColumnView(non_zero_index);
                Vec old_vec = errorHessian.getColumn(i);
                int non_zero_index_2 = 0;
                for (int j = 0; j < totalNumOfVariables; j++) {
                    if (!isColumnEmpty[j]) {
                        new_vec.set(non_zero_index_2++, old_vec.get(j));
                    }
                }
                non_zero_index++;
            }
        }

        errorHessian = new_hessian;
        return new_derivative;
    }

    private double getProbabilityVolume(ArrayList<Outcome> outcomes) {
        return outcomes.stream().mapToDouble(outcome -> outcome.probability).sum();
    }

    private void computeErrorOfOutput(boolean print_forward) {
        for (int time = inputs.length - 1; time > 0; time--) {
            for (int i = 0; i < outputNodes.size(); i++) {
                computeErrorOfOutput(outputNodes.get(i), time, targets[time][i]);
            }
        }

        assert Double.isFinite(total_error.getAverage());
        if (print_forward) {
            System.out.println(total_error.getAverage());
        }
        total_error.reset();
    }

    private void computeErrorOfOutput(OutputNode node, int timestep, Double target) {
        ArrayList<Outcome> outcomes = networkHistory.getStateOfNode(timestep, node);
        if (outcomes == null) {
            return;
        }

        if (target == null) {
            for (Outcome outcome : outcomes) {
                outcome.passRate.add(0, 1);
            }
            return;
        }

        double probabilityVolume = getProbabilityVolume(outcomes);
        if (probabilityVolume == 0) {
            return;
        }

        for (Outcome outcome : outcomes) {
            // if (timestep == this.timestep) {
            // outcome.passRate.add(1 - 1d / timestep, 1);
            // } else {
            outcome.passRate.add(1, 1);
            // }

            double error_derivative = errorFunction.error_derivative(outcome.activatedValue, target);
            double error_second_derivative = errorFunction.error_second_derivative(outcome.activatedValue, target);
            double prob = outcome.probability / probabilityVolume;

            // accumulate jacobians
            errorDerivative.mutableAdd(outcome.errorJacobian.multiply(prob * error_derivative));

            // accumulate hessian
            Matrix JJT = outcome.errorJacobian.multiplyTranspose(outcome.errorJacobian);
            errorHessian.mutableAdd(JJT.multiply(prob * error_second_derivative));
            errorHessian.mutableAdd(outcome.errorHessian.multiply(prob * error_derivative));

            double error = errorFunction.error(outcome.activatedValue, target);
            // assert Double.isFinite(errorFunction.error(outcome.activatedValue, target));
            total_error.add(error, outcome.probability);
        }

    }

    private void backpropagateErrors() {
        ArrayList<INode> nodes = network.getNodes();
        while (timestep >= 0) {
            HashMap<INode, ArrayList<Outcome>> state = networkHistory.getStateAtTimestep(timestep);
            for (Entry<INode, ArrayList<Outcome>> e : state.entrySet()) {
                INode node = nodes.get(e.getKey().getID());
                updateNodeForTimestep((ITrainable) node, e.getValue());
            }
            timestep--;
        }
    }

    private void updateNodeForTimestep(ITrainable node, ArrayList<Outcome> outcomesAtTIme) {
        prepareOutputDistributionAdjustments(node, outcomesAtTIme);
        outcomesAtTIme.forEach(outcome -> adjustProbabilitiesForOutcome(node, outcome));
    }

    private void applyErrorSignals() {
        parameterDeltas.mutableMultiply(epsilon);
        allNodes.forEach(this::applyErrorSignalsToNode);
        allNodes.forEach(ITrainable::applyDistributionUpdate);
        allNodes.forEach(ITrainable::applyFilterUpdate);
    }

    private void applyInputToNode(HashMap<Integer, ? extends IInputNode> inputNodeMap) {
        applyInputToNode(inputNodeMap, inputs, timestep);
    }

    /**
     * Use the outcomes to prepare weighted adjustments to the outcome distribution
     */
    public void prepareOutputDistributionAdjustments(ITrainable node, ArrayList<Outcome> allOutcomes) {
        IExpectationAdjuster adjuster = node.getOutputDistributionAdjuster();
        for (Outcome o : allOutcomes) {
            // weigh the outcomes by their probability of occurring
            adjuster.prepareAdjustment(o.probability, new double[] { o.activatedValue });
        }
    }

    public void adjustProbabilitiesForOutcome(ITrainable node, Outcome outcome) {
        if (!outcome.passRate.hasValues() || outcome.probability == 0) {
            return;
        }
        double pass_rate = outcome.passRate.getAverage();

        // Add another point for the net firing chance distribution
        IExpectationAdjuster adjuster = node.getSignalChanceDistributionAdjuster();
        adjuster.prepareAdjustment(outcome.probability, new double[] { pass_rate });

        // Reinforce the filter with the pass rate for each point
        for (int i = 0; i < outcome.sourceNodes.length; i++) {
            INode sourceNode = outcome.sourceNodes[i];

            Arc arc = node.getIncomingConnectionFrom(sourceNode).get(); // should be guaranteed to exist

            // if the error is not defined and the pass rate is 0, then zero error should be
            // expected
            if (arc.filterAdjuster != null) {
                double shifted_value = outcome.sourceOutcomes[i].activatedValue;
                double prob = outcome.probability * arc.filter.getChanceToSend(shifted_value)
                        / outcome.sourceTransferProbabilities[i];
                arc.filterAdjuster.prepareAdjustment(prob, new double[] { shifted_value, pass_rate });
            }
        }

    }

    public static void applyInputToNode(HashMap<Integer, ? extends IInputNode> inputNodeMap, Double[][] input,
            int counter) {
        InputNode[] sortedNodes = inputNodeMap.values().stream().sorted().toArray(InputNode[]::new);

        for (int i = 0; i < sortedNodes.length; i++) {
            if (input[counter][i] != null) {
                sortedNodes[i].acceptUserForwardSignal(input[counter][i]);
            }
        }
    }

    public void applyErrorSignalsToNode(ITrainable node) {

        double[] allDeltas = parameterDeltas.getColumn(0).arrayCopy();
        double[] gradient = gradientOfNode(node, allDeltas);
        node.applyDelta(gradient);
    }

    private double[] gradientOfNode(ITrainable node, double[] allDeltas) {
        int startIdx = vectorNodeOffset.get(node);
        int length = node.getNumberOfVariables();
        double[] gradient = new double[length];
        System.arraycopy(allDeltas, startIdx, gradient, 0, length);
        return gradient;
    }

}
