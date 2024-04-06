package com.lucasbrown.GraphNetwork.Local.Nodes;

public interface IInputNode {

    public abstract void acceptUserInferenceSignal(double value);
    public abstract void acceptUserForwardSignal(double value);
}
