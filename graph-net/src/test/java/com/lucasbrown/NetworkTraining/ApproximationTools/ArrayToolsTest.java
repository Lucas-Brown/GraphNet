package com.lucasbrown.NetworkTraining.ApproximationTools;

import static org.junit.Assert.assertEquals;

import java.util.HashSet;
import java.util.List;

import org.junit.Test;

public class ArrayToolsTest {

    private static final Double d0 = Double.valueOf(0);
    private static final Double d1 = Double.valueOf(1);
    private static final Double d2 = Double.valueOf(2);
    private static final Double d3 = Double.valueOf(3);
    private static final Double d4 = Double.valueOf(4);
    private static final Double d5 = Double.valueOf(5);

    @Test
    public void testFlatCartesianPowerProduct() {
        HashSet<HashSet<Double>> setSet = new HashSet<>();
        setSet.add(new HashSet<>(List.of(d0, d1)));
        setSet.add(new HashSet<>(List.of(d2, d3)));

        HashSet<HashSet<Double>> expected = new HashSet<>();
        expected.add(new HashSet<>(0));
        expected.add(new HashSet<>(List.of(d0)));
        expected.add(new HashSet<>(List.of(d2)));
        expected.add(new HashSet<>(List.of(d0, d2)));
        expected.add(new HashSet<>(List.of(d3)));
        expected.add(new HashSet<>(List.of(d0, d3)));
        expected.add(new HashSet<>(List.of(d1)));
        expected.add(new HashSet<>(List.of(d1, d2)));
        expected.add(new HashSet<>(List.of(d1, d3)));

        HashSet<HashSet<Double>> powProduct = IterableTools.flatCartesianPowerProduct(setSet);

        assertEquals(expected, powProduct);
    }

    @Test
    public void testFlatCartesianProduct() {
        HashSet<HashSet<Double>> setSet = new HashSet<>();
        setSet.add(new HashSet<>(List.of(d0, d1)));
        setSet.add(new HashSet<>(List.of(d2)));
        setSet.add(new HashSet<>(List.of(d3, d4, d5)));

        HashSet<HashSet<Double>> expected = new HashSet<>();
        expected.add(new HashSet<>(List.of(d0, d2, d3)));
        expected.add(new HashSet<>(List.of(d0, d2, d4)));
        expected.add(new HashSet<>(List.of(d0, d2, d5)));
        expected.add(new HashSet<>(List.of(d1, d2, d3)));
        expected.add(new HashSet<>(List.of(d1, d2, d4)));
        expected.add(new HashSet<>(List.of(d1, d2, d5)));

        HashSet<HashSet<Double>> product = IterableTools.flatCartesianProduct(setSet);
     
        assertEquals(expected, product);
    }

    @Test
    public void testFlatCartesianProduct2() {

        HashSet<HashSet<Double>> set1 = new HashSet<>();
        set1.add(new HashSet<>(List.of(d0)));
        set1.add(new HashSet<>(List.of(d1)));

        HashSet<Double> set2 = new HashSet<>(List.of(d2, d3));

        HashSet<HashSet<Double>> expected = new HashSet<>();
        expected.add(new HashSet<>(List.of(d0, d2)));
        expected.add(new HashSet<>(List.of(d0, d3)));
        expected.add(new HashSet<>(List.of(d1, d2)));
        expected.add(new HashSet<>(List.of(d1, d3)));

        HashSet<HashSet<Double>> product = IterableTools.flatCartesianProduct(set1, set2);
     
        assertEquals(expected, product);
    }

    @Test
    public void testPowerSet() {
        HashSet<Double> set = new HashSet<>(List.of(d0, d1, d2));

        HashSet<HashSet<Double>> truePowerSet = new HashSet<>();
        truePowerSet.add(new HashSet<>(0));
        truePowerSet.add(new HashSet<>(List.of(d0)));
        truePowerSet.add(new HashSet<>(List.of(d1)));
        truePowerSet.add(new HashSet<>(List.of(d2)));
        truePowerSet.add(new HashSet<>(List.of(d0, d1)));
        truePowerSet.add(new HashSet<>(List.of(d0, d2)));
        truePowerSet.add(new HashSet<>(List.of(d1, d2)));
        truePowerSet.add(new HashSet<>(List.of(d0, d1, d2)));

        HashSet<HashSet<Double>> powerSet = IterableTools.powerSet(set);

        assertEquals(truePowerSet, powerSet);
    }
}
