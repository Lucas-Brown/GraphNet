package com.lucasbrown.GraphNetwork.Global.Trainers;


public interface ITrainer {

    /**
     * input and target dimension : [timestep][node]
     * 
     * @param inputs
     * @param targets
     */
    public void setTrainingData(Double[][] inputs, Double[][] targets);
    public void trainNetwork(int steps, int print_interval);
}
