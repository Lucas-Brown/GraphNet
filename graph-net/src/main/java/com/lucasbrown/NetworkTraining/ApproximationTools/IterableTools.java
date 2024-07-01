package com.lucasbrown.NetworkTraining.ApproximationTools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class IterableTools {

    public static double[] slice(double[] dArr, int start, int count){
        double[] slice = new double[count];
        System.arraycopy(dArr, start, slice, 0, count);
        return slice;
    }

    public static final double[][] array2DCopy(double[][] toCopy) {
        final double[][] copy = new double[toCopy.length][];
        for (int i = 0; i < copy.length; i++) {
            copy[i] = toCopy[i].clone(); // clone is safe for value-types
        }
        return copy;
    }

    public static final int[] union(int[] arr1, int[] arr2) {
        HashSet<Integer> union = new HashSet<Integer>(arr1.length + arr2.length);
        for (int i : arr1) {
            union.add(i);
        }
        for (int i : arr2) {
            union.add(i);
        }
        return union.stream().mapToInt(i -> i).toArray();
    }

    public static final int[] concatenate(int[] arr1, int[] arr2) {
        int[] concat = Arrays.copyOf(arr1, arr1.length + arr2.length);
        System.arraycopy(arr2, 0, concat, arr1.length, arr2.length);
        return concat;
    }

    public static <T> boolean hasIntersection(HashSet<T> s1, HashSet<T> s2) {
        HashSet<T> intersection = new HashSet<T>(s1);
        intersection.retainAll(s2);
        return intersection.size() > 0;
    }

    public static double[] applyMask(double[] array, int mask){
        double[] filtered_array = new double[Integer.bitCount(mask)];
        int filtered_count = 0;
        for(int i = 0; i < array.length; i++){
            if(((mask >> i) & 0b1) == 1){
                filtered_array[filtered_count++] = array[i]; 
            }
        }
        return filtered_array;
    }

    /**
     * Creates a shallow copy of a set of sets.
     * 
     * @param <T>    the type of elements in the sets
     * @param toCopy the set of sets to copy
     * @return a shallow copy of the input set
     */
    @SuppressWarnings("unchecked")
    public static <T> HashSet<HashSet<T>> shallowCopy(HashSet<HashSet<T>> toCopy) {
        // Create a new set to store the copied sets
        HashSet<HashSet<T>> copy = new HashSet<>(toCopy.size());
        // Iterate over each set in the input set
        for (HashSet<T> t : toCopy) {
            // Clone each set and add it to the copy set
            copy.add((HashSet<T>) t.clone());
        }
        return copy;
    }

    /**
     * Generates the Cartesian product of a collection of sets, then generates the
     * power set of each product, and finally pairs each subset with its original set.
     * 
     * @param <T>       the type of elements in the sets
     * @param collection the collection of sets
     * @return a set of pairs of subsets and their original sets
     */
    public static <T> HashSet<Pair<HashSet<T>, HashSet<T>>> flatCartesianPowerProductPair(Collection<? extends Collection<T>> collection) {
        // Generate the Cartesian product of the input collection
        HashSet<HashSet<T>> cartesianProduct = flatCartesianProduct(collection);
        // Initialize a set to store the result
        HashSet<Pair<HashSet<T>, HashSet<T>>> flattened = new HashSet<>();

        // Iterate over each set in the Cartesian product
        for (HashSet<T> set : cartesianProduct) {
            // Generate the power set of the current set
            HashSet<HashSet<T>> powerSet = powerSet(set);
            // Map each subset to a pair with its original set and add to the result
            flattened.addAll(powerSet.stream().map(powSet -> new Pair<>(powSet, set)).toList());
        }
        return flattened;
    }

    /**
     * Generates the Cartesian product of a collection of sets, then generates the
     * power set of each product.
     * 
     * @param <T>       the type of elements in the sets
     * @param collection the collection of sets
     * @return a set of subsets of the Cartesian product
     */
    public static <T> HashSet<HashSet<T>> flatCartesianPowerProduct(Collection<? extends Collection<T>> collection) {
        // Generate the Cartesian product of the input collection
        HashSet<HashSet<T>> cartesianProduct = flatCartesianProduct(collection);
        // Initialize a set to store the result
        HashSet<HashSet<T>> flattened = new HashSet<>();

        // Iterate over each set in the Cartesian product
        for (HashSet<T> set : cartesianProduct) {
            // Generate the power set of the current set and add to the result
            flattened.addAll(powerSet(set));
        }
        return flattened;
    }

    /**
     * Generates the power set of a given set.
     * 
     * @param <T> the type of elements in the set
     * @param set the input set
     * @return the power set of the input set
     */
    public static <T> HashSet<HashSet<T>> powerSet(Set<T> set) {
        // Initialize the power set with an empty set
        HashSet<HashSet<T>> powerSet = new HashSet<>();
        powerSet.add(new HashSet<>());

        // Iterate over each element in the input set
        for (T t : set) {
            // Create a copy of the current power set
            HashSet<HashSet<T>> copy = shallowCopy(powerSet);
            // Add the current element to each set in the copy
            for (HashSet<T> tList : copy) {
                tList.add(t);
            }
            // Add the modified sets to the power set
            powerSet.addAll(copy);
        }
        return powerSet;
    }

    /**
     * Generates the Cartesian product of a collection of sets.
     * 
     * @param <T>       the type of elements in the sets
     * @param collection the collection of sets
     * @return the Cartesian product of the input sets
     */
    public static <T> HashSet<HashSet<T>> flatCartesianProduct(Collection<? extends Collection<T>> collection) {
        // Initialize the Cartesian product
        HashSet<HashSet<T>> cartesianProduct = new HashSet<>();

        // If the input collection is empty, return an empty set
        if (collection.isEmpty())
            return cartesianProduct;

        // Get the iterator for the input collection
        Iterator<? extends Collection<T>> setItr = collection.iterator();
        // Initialize the Cartesian product with the first set in the collection
        for (T t : setItr.next()) {
            HashSet<T> singleton = new HashSet<>(1);
            singleton.add(t);
            cartesianProduct.add(singleton);
        }

        // Iterate over the remaining sets in the collection
        while (setItr.hasNext()) {
            // Update the Cartesian product by taking the product with the next set
            cartesianProduct = flatCartesianProduct(cartesianProduct, setItr.next());
        }
        return cartesianProduct;
    }

    /**
     * Generates the Cartesian product of two sets of sets.
     * 
     * @param <T> the type of elements in the sets
     * @param s1  the first set of sets
     * @param s2  the second set
     * @return the Cartesian product of the input sets
     */
    public static <T> HashSet<HashSet<T>> flatCartesianProduct(HashSet<? extends Collection<T>> s1, Collection<T> s2) {
        // Initialize the Cartesian product
        HashSet<HashSet<T>> cartesianProduct = new HashSet<>();

        // Iterate over each set in the first input set
        for (Collection<T> arr : s1) {
            // Iterate over each element in the second input set
            for (T t : s2) {
                // Create a copy of the current set from the first input set
                HashSet<T> sCopy = new HashSet<>(arr);
                // Add the current element from the second input set to the copy
                sCopy.add(t);
                // Add the modified set to the Cartesian product
                cartesianProduct.add(sCopy);
            }
        }
        return cartesianProduct;
    }
}
