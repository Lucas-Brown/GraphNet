package com.lucasbrown.GraphNetwork.Global;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.IInputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.IOutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.History;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;

public class BackpropTrainer {
    
    public double epsilon = 0.01;

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
                System.out.println(network);
            }
            networkHistory.captureState();
        }
        timestep--;
    }


    private void computeErrorOfOutputs(boolean print_forward){
        double total_error = 0;
        for(int time = timestep; time > 0; time--){
            for (int i = 0; i < outputNodes.size(); i++) {
                total_error += computeErrorOfOutput(outputNodes.get(i), time, targets[time][i]);
            }
        }
        if(print_forward){
            System.out.println(total_error);
        }
    }

    private double computeErrorOfOutput(OutputNode node, int timestep, double target){
        ArrayList<Outcome> outcomes = networkHistory.getStateOfNode(timestep, node.getID());

        // double normalization_const = 0;
        // for(Outcome outcome : outcomes){
        //     normalization_const += outcome.probability;
        // }

        double total_error = 0;
        for(Outcome outcome : outcomes){
            // double error = outcome.probability * errorFunction.error_derivative(outcome.netValue, target) / normalization_const;
            total_error += errorFunction.error(outcome.activatedValue, target);
            double error = errorFunction.error_derivative(outcome.activatedValue, target);
            node.recieveError(timestep, outcome.binary_string, error);
        }
        return total_error;
    }

    private void backpropagateErrors() {
        ArrayList<INode> nodes = network.getNodes();
        while(timestep > 0){
            HashMap<Integer, ArrayList<Outcome>> state = networkHistory.getStateAtTimestep(timestep);
            for(Entry<Integer, ArrayList<Outcome>> e : state.entrySet()){
                INode node = nodes.get(e.getKey());
                node.sendErrorsBackwards(e.getValue(), timestep);
            }
            timestep--;
        }
    }

    private void applyErrorSignals(){
        network.getNodes().forEach(this::applyErrorSignalsToNode);
    }

    private void applyErrorSignalsToNode(INode node){
        node.applyErrorSignals(epsilon);
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
