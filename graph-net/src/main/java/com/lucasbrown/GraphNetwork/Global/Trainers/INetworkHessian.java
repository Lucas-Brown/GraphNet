package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.NetworkTraining.History;

import jsat.linear.Matrix;

public interface INetworkHessian extends INetworkGradient {
    
    public ArrayList<HashMap<Outcome, Matrix>> getHessian(History<Outcome, INode> networkHistory);
}
