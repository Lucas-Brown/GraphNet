package com.lucasbrown.NetworkTraining.ApproximationTools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

public class ArrayTools {

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

    @SuppressWarnings("unchecked")
    public static <T> HashSet<HashSet<T>> shallowCopy(HashSet<HashSet<T>> toCopy) {
        HashSet<HashSet<T>> copy = new HashSet<>(toCopy.size());
        for (HashSet<T> t : toCopy) {
            copy.add((HashSet<T>) t.clone());
        }
        return copy;
    }

    public static <T> HashSet<Pair<HashSet<T>, HashSet<T>>> flatCartesianPowerProductPair(Collection<? extends Collection<T>> collection) {
        HashSet<Pair<HashSet<T>, HashSet<T>>> flattened = new HashSet<>();
        HashSet<HashSet<T>> cartesianProduct = flatCartesianProduct(collection);
        for (HashSet<T> set : cartesianProduct) {
            flattened.addAll(powerSet(set).stream().map(powSet -> new Pair<>(powSet, set)).toList());
        }
        return flattened;
    }

    public static <T> HashSet<HashSet<T>> flatCartesianPowerProduct(Collection<? extends Collection<T>> collection) {
        HashSet<HashSet<T>> flattened = new HashSet<>();
        HashSet<HashSet<T>> cartesianProduct = flatCartesianProduct(collection);
        for (HashSet<T> set : cartesianProduct) {
            flattened.addAll(powerSet(set));
        }
        return flattened;
    }

    public static <T> HashSet<HashSet<T>> powerSet(Set<T> set) {
        HashSet<HashSet<T>> powerSet = new HashSet<>();
        powerSet.add(new HashSet<>());

        for (T t : set) {
            HashSet<HashSet<T>> copy = shallowCopy(powerSet);
            for (HashSet<T> t_list : copy) {
                t_list.add(t);
            }
            powerSet.addAll(copy);
        }
        return powerSet;
    }

    public static <T> HashSet<HashSet<T>> flatCartesianProduct(Collection<? extends Collection<T>> collection) {
        HashSet<HashSet<T>> cartesianProduct = new HashSet<>();
        if (collection.isEmpty())
            return cartesianProduct;

        Iterator<? extends Collection<T>> setItr = collection.iterator();
        for (T t : setItr.next()) {
            HashSet<T> singleton = new HashSet<>(1);
            singleton.add(t);
            cartesianProduct.add(singleton);
        }

        while(setItr.hasNext())
        {
            cartesianProduct = flatCartesianProduct(cartesianProduct, setItr.next());
        }
        return cartesianProduct;
    }

    public static <T> HashSet<HashSet<T>> flatCartesianProduct(HashSet<? extends Collection<T>> s1, Collection<T> s2) {
        HashSet<HashSet<T>> cartesianProduct = new HashSet<>();
        for (Collection<T> arr : s1) {
            for (T t : s2) {
                HashSet<T> s_copy = new HashSet<>(arr);

                s_copy.add(t);
                cartesianProduct.add(s_copy);
            }
        }
        return cartesianProduct;
    }

}
