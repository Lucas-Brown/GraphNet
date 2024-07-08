package com.lucasbrown.GraphNetwork.Global;

public class IncompleteNodeException extends RuntimeException {

    public IncompleteNodeException(String message) {
        super(message);
    }

    public IncompleteNodeException() {
        super("Node could not be built with the provided parameters. Please check that all node parameters have been assigned values");
    }
}
