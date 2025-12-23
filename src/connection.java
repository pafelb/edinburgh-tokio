public class connection {
    int id;
    boolean boat;
    int city1, city2;
    int building_cost;
    int color,color2;
    int reward;
    boolean segmented;

    public connection(boolean boat, int id, int building_cost, int color, int color2, int reward, boolean segmented) {
        this.boat = boat;
        this.id = id;
        this.building_cost = building_cost;
        this.color = color;
        this.color2 = color2;
        this.reward = reward;
        this.segmented = segmented;
    }
    // 0 white, 1 green, 2 red, 3 black, 4 pink, 5 yellow, 6 segmented
}
