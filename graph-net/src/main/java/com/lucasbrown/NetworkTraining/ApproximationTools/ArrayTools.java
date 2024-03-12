package com.lucasbrown.NetworkTraining.ApproximationTools;

import java.util.Arrays;
import java.util.HashSet;

public class ArrayTools {
    
    public static final double[][] array2DCopy(double[][] toCopy){
        final double[][] copy = new double[toCopy.length][];
        for (int i = 0; i < copy.length; i++) {
            copy[i] = toCopy[i].clone(); // clone is safe for value-types
        }
        return copy;
    }

    public static final int[] union(int[] arr1, int[] arr2){
        HashSet<Integer> union = new HashSet<Integer>(arr1.length + arr2.length);
        for (int i : arr1) {
            union.add(i);
        }
        for (int i : arr2) {
            union.add(i);
        }
        return union.stream().mapToInt(i -> i).toArray();
    }

    public static final int[] concatenate(int[] arr1, int[] arr2)
    {
        int[] concat = Arrays.copyOf(arr1, arr1.length + arr2.length);
        System.arraycopy(arr2, 0, concat, arr1.length, arr2.length);
        return concat;        
    }
}
