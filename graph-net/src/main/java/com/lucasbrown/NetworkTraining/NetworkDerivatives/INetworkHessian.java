package com.lucasbrown.NetworkTraining.NetworkDerivatives;

import java.util.ArrayList;
import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.NetworkTraining.History.NetworkHistory;

import jsat.linear.Matrix;

public interface INetworkHessian extends INetworkGradient {
    
    public ArrayList<HashMap<Outcome, Matrix>> getHessian(NetworkHistory networkHistory);
}
