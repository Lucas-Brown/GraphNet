

import static org.junit.Assert.assertEquals;

import java.util.function.ToDoubleBiFunction;

import org.junit.Test;

import com.lucasbrown.NetworkTraining.LinearInterpolation2D;
import com.lucasbrown.NetworkTraining.LinearRange;
import com.lucasbrown.NetworkTraining.Range;

public class InterpolationTest {
    
    private final double tollerance = 1E-6;
    private final Range x_range = new LinearRange(0, 1, 4, true, true);
    private final Range y_range = new LinearRange(0, 1, 4, true, true);

    @Test
    public void testInterp()
    {
        ToDoubleBiFunction<Double, Double> linearGenerator = (x, y) -> x+y;
        LinearInterpolation2D lerp2D = new LinearInterpolation2D(x_range, y_range, linearGenerator);

        // interpolation
        assertEquals(0, lerp2D.interpolate(0,0), tollerance);
        assertEquals(0.5, lerp2D.interpolate(0.5,0), tollerance);
        assertEquals(0.3, lerp2D.interpolate(0,0.3), tollerance);
        assertEquals(1, lerp2D.interpolate(0.5,0.5), tollerance);
        assertEquals(2, lerp2D.interpolate(1, 1), tollerance);

        // exact extrapolation
        assertEquals(-3, lerp2D.interpolate(-1, -2), tollerance);
    }
}
