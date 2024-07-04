package com.lucasbrown.GraphNetwork.Global.Trainers;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.NetworkTraining.History;

import jsat.linear.Vec;

public interface IGradient {

    public Vec computeGradient(History<Outcome, INode> networkHistory);

    public double getTotalError(History<Outcome, INode> networkHistory);

    public void setTargets(Double[][] targets); 
    public Double[][] getTargets();
}
