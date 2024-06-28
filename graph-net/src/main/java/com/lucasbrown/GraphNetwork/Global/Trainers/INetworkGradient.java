package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.NetworkTraining.History;

import jsat.linear.Vec;

public interface INetworkGradient {
    
    public ArrayList<HashMap<Outcome, Vec>> getGradient(History<Outcome, INode> networkHistory);

}
