package com.lucasbrown.GraphNetwork.Global;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map.Entry;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.IInputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.History;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.WeightedAverage;

public class BackpropTrainer {
    
    public double epsilon = 0.01;
    private WeightedAverage total_error;

    private int timestep;
    private final GraphNetwork network;
    private final History networkHistory;
    private final ErrorFunction errorFunction;

    private Double[][] inputs;
    private Double[][] targets;

    private ArrayList<OutputNode> outputNodes;

    public BackpropTrainer(GraphNetwork network, ErrorFunction errorFunction)
    {
        this.network = network;
        this.errorFunction = errorFunction;
        networkHistory = new History(network);

        network.setInputOperation(this::applyInputToNode);
        outputNodes = network.getOutputNodes();

        total_error = new WeightedAverage();
    }

    /**
     * input and target dimension : [timestep][node]
     * @param inputs
     * @param targets
     */
    public void setTrainingData(Double[][] inputs, Double[][] targets){
        this.inputs = inputs;
        this.targets = targets;
    }

    public void trainNetwork(int steps, int print_interval){
        while(steps-- > 0){
            trainingStep(steps % print_interval == 0);
        }
    }

    public void trainingStep(boolean print_forward){
        captureForward(print_forward);

        computeErrorOfOutputs(print_forward);
        backpropagateErrors();
        applyErrorSignals();
        network.deactivateAll();
        networkHistory.burnHistory();
    }


    private void captureForward(boolean print_forward){
        for (timestep = 0; timestep < inputs.length; timestep++) {
            network.trainingStep();
            if(print_forward) {
                System.out.println(network.toString() + " | Target = " + Arrays.toString(targets[timestep]));
            }
            networkHistory.captureState();
        }
        timestep--;
    }


    private void computeErrorOfOutputs(boolean print_forward){
        for(int time = timestep; time > 0; time--){
            for (int i = 0; i < outputNodes.size(); i++) {
                computeErrorOfOutput(outputNodes.get(i), time, targets[time][i]);
            }
        }
        //assert total_error.getAverage() < 1E6;
        assert Double.isFinite(total_error.getAverage());
        if(print_forward){
            System.out.println(total_error.getAverage());
        }
        total_error.reset();
    }

    private void computeErrorOfOutput(OutputNode node, int timestep, Double target){
        ArrayList<Outcome> outcomes = networkHistory.getStateOfNode(timestep, node.getID());
        if(outcomes == null){
            return;
        }

        if(target == null){
            for(Outcome outcome : outcomes){
                outcome.passRate.add(0, 1);
            }
            return;
        }

        for(Outcome outcome : outcomes){
            outcome.passRate.add(1, 1);
            outcome.errorOfOutcome.add(errorFunction.error_derivative(outcome.activatedValue, target), outcome.probability);
            assert Double.isFinite(errorFunction.error(outcome.activatedValue, target));
            total_error.add(errorFunction.error(outcome.activatedValue, target), outcome.probability);
        }
        
    }

    private void backpropagateErrors() {
        ArrayList<INode> nodes = network.getNodes();
        while(timestep >= 0){
            HashMap<Integer, ArrayList<Outcome>> state = networkHistory.getStateAtTimestep(timestep);
            for(Entry<Integer, ArrayList<Outcome>> e : state.entrySet()){
                updateNodeForTimestep(nodes.get(e.getKey()), e.getValue());
            }
            timestep--;
        }
    }

    private void updateNodeForTimestep(INode node, ArrayList<Outcome> outcomesAtTIme){
        node.prepareOutputDistributionAdjustments(outcomesAtTIme);
        outcomesAtTIme.forEach(node::sendErrorsBackwards);
        outcomesAtTIme.forEach(node::adjustProbabilitiesForOutcome);
    }

    private void applyErrorSignals(){
        network.getNodes().forEach(this::applyErrorSignalsToNode);
        network.getNodes().forEach(INode::applyDistributionUpdate);
        network.getNodes().forEach(INode::applyFilterUpdate);
    }

    private void applyErrorSignalsToNode(INode node){
        node.applyErrorSignals(epsilon, networkHistory.getHistoryOfNode(node.getID()));
    }

    private void applyInputToNode(HashMap<Integer, ? extends IInputNode> inputNodeMap){
        applyInputToNode(inputNodeMap, inputs, timestep);
    }

    public static void applyInputToNode(HashMap<Integer, ? extends IInputNode> inputNodeMap, Double[][] input, int counter){
        InputNode[] sortedNodes = inputNodeMap.values().stream().sorted().toArray(InputNode[]::new);

        for (int i = 0; i < sortedNodes.length; i++) {
            if(input[counter][i] != null){
                sortedNodes[i].acceptUserForwardSignal(input[counter][i]);
            }
        }
    }
}
