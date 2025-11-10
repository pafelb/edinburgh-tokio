import java.util.Arrays;

public class map {
    public int[] strecken = new int[129];

    public map() {
        Arrays.fill(strecken, 0);
    }

    private void build(int track, int player) {
        if (strecken[track] == 0)
            strecken[track] = player;
    }


}
