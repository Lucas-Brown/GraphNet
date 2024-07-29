package com.lucasbrown.NetworkTraining.Trainers;

import java.util.ArrayList;
import java.util.stream.Collectors;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.IOutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.ITrainableValueCombinator;
import com.lucasbrown.NetworkTraining.History.NetworkHistory;
import com.lucasbrown.NetworkTraining.NetworkDerivatives.ForwardFilterGradient;
import com.lucasbrown.NetworkTraining.NetworkDerivatives.ForwardNetworkFilterGradient;
import com.lucasbrown.NetworkTraining.NetworkDerivatives.ForwardNetworkGradient;
import com.lucasbrown.NetworkTraining.OutputDerivatives.CompleteNetworkGradient;
import com.lucasbrown.NetworkTraining.OutputDerivatives.DirectNetworkGradient;
import com.lucasbrown.NetworkTraining.OutputDerivatives.ErrorFunction;
import com.lucasbrown.NetworkTraining.OutputDerivatives.IGradient;
import com.lucasbrown.NetworkTraining.OutputDerivatives.OutcomeChanceFilterGradient;
import com.lucasbrown.NetworkTraining.OutputDerivatives.WeightedOutcomeChanceFilterGradient;
import com.lucasbrown.NetworkTraining.Solvers.ADAMSolver;
import com.lucasbrown.NetworkTraining.Solvers.ISolver;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class Trainer implements ITrainer {

    private final GraphNetwork network;

    public final WeightsLinearizer weightLinearizer;
    public final FilterLinearizer filterLinearizer;
    public final NetworkInputEvaluater networkEvaluater;
    public final IGradient weightsGradient;
    public final IGradient probabilityGradient;
    public final ISolver weightsSolver;
    public final ISolver probabilitySolver;

    private Vec weightsDeltas;
    private Vec probabilityDeltas;

    protected Double[][][] inputs;
    protected Double[][][] targets;

    public Trainer(NetworkInputEvaluater networkEvaluater, IGradient weightsGradient, ISolver weightsSolver,
            IGradient probabilityGradient, ISolver probabilitySolver, WeightsLinearizer weightLinearizer,
            FilterLinearizer filterLinearizer) {

        this.weightsGradient = weightsGradient;
        this.probabilityGradient = probabilityGradient;
        this.weightsSolver = weightsSolver;
        this.probabilitySolver = probabilitySolver;

        this.weightLinearizer = weightLinearizer;
        this.filterLinearizer = filterLinearizer;
        this.networkEvaluater = networkEvaluater;

        network = networkEvaluater.network;
    }

    /**
     * input and target dimension : [timestep][node]
     * 
     * @param inputs
     * @param targets
     */
    public void setTrainingData(Double[][] inputs, Double[][] targets) {
        setTrainingData(new Double[][][] { inputs }, new Double[][][] { targets });
    }

    /**
     * input and target dimension : [timestep][node]
     * 
     * @param inputs
     * @param targets
     */
    public void setTrainingData(Double[][][] inputs, Double[][][] targets) {
        this.inputs = inputs;
        this.targets = targets;
    }

    public void trainNetwork(int steps, int print_interval) {
        while (steps-- > 0) {
            trainingStep(steps % print_interval == 0);
        }
    }

    public void trainingStep(boolean print_forward) {
        NetworkHistory[] histories = computeAllHistories();
        if (print_forward) {
            printNetwork(histories);
        }

        Vec weightsGradient = aggregateWeightGradients(histories);
        Vec probabilityGradient = aggregateProbabilityGradients(histories);

        weightsDeltas = weightsSolver.solve(weightsGradient);
        probabilityDeltas = probabilitySolver.solve(probabilityGradient);

        applyWeightDeltas();
        applyProbabilityDeltas();
    }

    public NetworkHistory[] computeAllHistories() {
        NetworkHistory[] histories = new NetworkHistory[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            networkEvaluater.setInputData(inputs[i]);
            histories[i] = networkEvaluater.computeNetworkInference();
            assert histories[i].getNumberOfTimesteps() == inputs[i].length;
        }
        return histories;
    }

    private Vec aggregateWeightGradients(NetworkHistory[] histories) {
        Vec gradient = new DenseVector(weightLinearizer.totalNumOfVariables);
        for (int i = 0; i < inputs.length; i++) {
            weightsGradient.setTargets(targets[i]);
            Vec grad = weightsGradient.computeGradient(histories[i]);
            gradient.mutableAdd(grad);
        }
        return gradient.divide(inputs.length);
    }

    private Vec aggregateProbabilityGradients(NetworkHistory[] histories) {
        Vec gradient = new DenseVector(filterLinearizer.totalNumOfVariables);
        for (int i = 0; i < inputs.length; i++) {
            probabilityGradient.setTargets(targets[i]);
            gradient.mutableAdd(probabilityGradient.computeGradient(histories[i]));
        }
        return gradient.divide(inputs.length);
    }

    private void printNetwork(NetworkHistory[] histories) {
        NetworkHistory history = histories[0]; // print an example
        int time_count = history.getNumberOfTimesteps();

        StringBuilder sb = new StringBuilder();
        ArrayList<INode> nodes = network.getNodes();

        for (int t = 0; t < time_count; t++) {
            sb.append("Time Step ");
            sb.append(t);
            sb.append("\n\t");

            int outIdx = 0;
            for (int i = 0; i < nodes.size(); i++) {
                INode node = nodes.get(i);
                ArrayList<Outcome> outcomes = history.getStateOfRecord(t, node);

                if (outcomes == null || outcomes.isEmpty()) {
                    continue;
                }

                sb.append(node.getName());
                sb.append(": [");
                sb.append(outcomes.stream()
                        .sorted(Outcome::descendingProbabilitiesComparator)
                        .limit(2)
                        .map(Object::toString)
                        .collect(Collectors.joining(",")));

                if (node instanceof IOutputNode) {
                    sb.append("] | target = ");
                    sb.append(targets[0][t][outIdx++]);
                    sb.append("\n\t");
                } else {
                    sb.append("]\n\t");
                }
            }
            sb.append("\n");
        }

        sb.append("Accuracy error : ");
        weightsGradient.setTargets(targets[0]);
        sb.append(getTotalError(histories, weightsGradient));
        sb.append("\nConsistency error : ");
        probabilityGradient.setTargets(targets[0]);
        sb.append(getTotalError(histories, probabilityGradient));
        sb.append("\n");
        System.out.println(sb.toString());
    }

    public double getTotalError(NetworkHistory[] histories, IGradient errorEvaluator) {
        double error = 0;
        for (int i = 0; i < histories.length; i++) {
            NetworkHistory history = histories[i];
            errorEvaluator.setTargets(targets[i]);
            assert history.getNumberOfTimesteps() == targets[i].length;
            error += errorEvaluator.getTotalError(history);
        }
        return error / histories.length;
    }

    private void applyWeightDeltas() {
        weightLinearizer.allNodes.forEach(this::applyErrorSignalsToNode);
    }

    private void applyErrorSignalsToNode(INode node) {
        double[] allDeltas = weightsDeltas.arrayCopy();
        double[] gradient = weightLinearizer.nodeSlice(node, allDeltas);
        ((ITrainableValueCombinator) node.getValueCombinator()).applyDelta(gradient);
    }

    private void applyProbabilityDeltas() {
        filterLinearizer.allFilters.forEach(this::applyParameterUpdate);
    }

    private void applyParameterUpdate(IFilter filter) {
        double[] gradient = filterLinearizer.filterSlice(filter, probabilityDeltas.arrayCopy());
        filter.applyAdjustableParameterUpdate(gradient);
    }

    public static Trainer getDefaultTrainer(GraphNetwork network) {
        WeightsLinearizer weightLinearizer = new WeightsLinearizer(network);
        FilterLinearizer filterLinearizer = new FilterLinearizer(network);
        NetworkInputEvaluater networkEvaluater = new NetworkInputEvaluater(network);

        ErrorFunction erf = new ErrorFunction.MeanSquaredError();

        // CompleteNetworkGradient netGradient = new CompleteNetworkGradient(network,
        // new ForwardNetworkGradient(weightLinearizer),
        // new ForwardNetworkFilterGradient(filterLinearizer), erf,
        // null,weightLinearizer.totalNumOfVariables);

        DirectNetworkGradient netGradient = new DirectNetworkGradient(network,
                new ForwardNetworkGradient(weightLinearizer), null, erf, weightLinearizer.totalNumOfVariables);

        ADAMSolver weightsSolver = new ADAMSolver(netGradient, weightLinearizer.totalNumOfVariables);

        WeightedOutcomeChanceFilterGradient filterGradient = new WeightedOutcomeChanceFilterGradient(network,
                new ForwardFilterGradient(filterLinearizer), null, erf, filterLinearizer.totalNumOfVariables);
        ADAMSolver filterSolver = new ADAMSolver(filterGradient, filterLinearizer.totalNumOfVariables);

        return new Trainer(networkEvaluater, netGradient, weightsSolver, filterGradient, filterSolver, weightLinearizer,
                filterLinearizer);
    }
}
