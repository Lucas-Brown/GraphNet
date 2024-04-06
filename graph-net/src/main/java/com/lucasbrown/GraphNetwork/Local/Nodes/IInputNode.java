package com.lucasbrown.GraphNetwork.Local;

public interface IInputNode {

    public abstract void acceptUserInferenceSignal(double value);
    public abstract void acceptUserForwardSignal(double value);
}
