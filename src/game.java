import java.util.Arrays;

public class game {
    private deck deck;
    public boolean isOver;
    private int[][] cards_players = new int[4][25];
    public boolean[][] tracks_ownership= new boolean[4][129];
    public boolean[] tickets = new boolean[62];

    public static void main(String[] args) {
        game game = new game();
    }

    public game() {
        setup();
        for (int i = 0; i != 4; i++) {
            System.out.println("PLayer: " + i);
            for (int j = 0; j != 6; j++) {
                System.out.println(converter(j) + "single: " + cards_players[i][j] + "; double: " + cards_players[i][j + 6]);
            }
            boolean running = true;
        }


    }

    private String converter(int i) {
        switch (i) {
            case 0:
                return "white ";
            case 1:
                return "green ";
            case 2:
                return "red ";
            case 3:
                return "black ";
            case 4:
                return "pink ";
            case 5:
                return "yellow ";
        }
        return null;
    }


    private void setup() {
        deck = new deck();
        for (int i = 0; i != 4; i++) {
            for (int j = 0; j != cards_players[i].length; j++) {
                Arrays.fill(cards_players[i],0);
            }
        }
        Arrays.fill(tracks_ownership,0);
        init_allocation();
    }

    private void init_allocation() {
        for (int i = 0; i != 28; i++) {
            cards_players[i % 4][deck.boats.pop()]++;
        }
        for (int i = 0; i != 12; i++) {
            cards_players[i % 4][deck.locos.pop()]++;
        }
    }
    public float[] toVector(){
        float[] normalized = new float[645];
        for (int i = 0;i!=4;i++){
            for (int j =0;j!=129;j++){
                normalized[i*129+j] = tracks_ownership[i][j] ? 1 : 0;
            }
        }
        //516



        return normalized;
    }
}
