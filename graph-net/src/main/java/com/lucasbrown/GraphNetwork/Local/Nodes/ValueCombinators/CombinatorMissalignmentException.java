package com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators;

/**
 * The {@link CombinatorMissalignmentException} is thrown anytime a
 * missalignment occurs between an incoming signal combination and a combination
 * operation.
 * 
 * For example, a method is given a set of signals and a binary key, but the
 * number 1's in the binary key does not correspond to the size of the signal
 * set. In this scenario, there is a missalignment between the signal set and
 * the binary key
 * 
 */
public class CombinatorMissalignmentException extends RuntimeException  {

    public CombinatorMissalignmentException() {
        super();
    }

    public CombinatorMissalignmentException(String message) {
        super(message);
    }

    public CombinatorMissalignmentException(String message, Throwable cause) {
        super(message, cause);
    }

    public CombinatorMissalignmentException(Throwable cause) {
        super(cause);
    }
}
