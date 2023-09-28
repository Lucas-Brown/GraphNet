package src.Tests;

import java.util.HashMap;

public class SetTest {
    
    public static void main(String[] args)
    {
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(1, 10);
        map.put(2, 20);
        map.put(3, 30);
        map.put(4, 40);

        map.values().remove(30);

        System.out.println(map.keySet());
    }
}
