package com.lucasbrown.GraphNetwork.Global.Trainers;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.NetworkTraining.History;

import jsat.linear.Vec;

public interface ISolver {

    public Vec solve(History<Outcome, INode> history);
}
