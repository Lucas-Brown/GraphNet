package com.lucasbrown.NetworkTraining;

/**
 * This exception should be thrown anytime the network structure is deemed untrainable.
 */
public class UntrainableNetworkException extends RuntimeException {
    
    public UntrainableNetworkException() {
        super();
    }

    public UntrainableNetworkException(String message) {
        super(message);
    }

    public UntrainableNetworkException(String message, Throwable cause) {
        super(message, cause);
    }

    public UntrainableNetworkException(Throwable cause) {
        super(cause);
    }
}
