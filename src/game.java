import java.util.Arrays;

public class game {
    private deck deck;
    private int remaining_rounds = 3;
    private int[][] cards_players = new int[4][25];
    private boolean[][] tracks_ownership = new boolean[4][129];
    private boolean[][] tickets = new boolean[4][62];
    private int[][] boats_locos_buildt = new int[4][4];
    private int[] lane = new int[6];
    private int current_player_move = 0;
    private gamestate gamestate;
    TicketToRideAgent agent = new TicketToRideAgent();


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


        }
        for (int i = 0; i != 6; i++) {
            System.out.print(deck.cardIdToString(lane[i]) + " ");
        }
        boolean running = true;
        while (running) {
            deck.shuffle();



            //endloop check for last round
            if (1 == 1) {//TODO later input value from action from agent, only activates if last action was build track

            }
            if (remaining_rounds != 3)
                remaining_rounds--;
        }

    }

    private void check_last_round() {
        if (boats_locos_buildt[current_player_move][0] * 2 + boats_locos_buildt[current_player_move][1] + boats_locos_buildt[current_player_move][2] + boats_locos_buildt[current_player_move][3] >= 54) {
            remaining_rounds = 2;
        }
    }

    private void draw_from_lane(int pick, boolean lay_open_boats) {   //TODO remember to illegal out if stack is empty later

        if (pick < 6) {
            cards_players[current_player_move][pick]++;
            if (lay_open_boats)
                lane[pick] = deck.boats.pop();
            else
                lane[pick] = deck.boats.pop();
        } else {
            int t;
            if (lay_open_boats) {
                t = deck.boats.pop();
            } else {
                t = deck.locos.pop();
            }
            cards_players[current_player_move][t]++;
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

    private void action(int action) {
        if (action <= 129) {


        }
    }

    private void buildConnection(int action) {

    }


    private void setup() {
        deck = new deck();
        for (int i = 0; i != 4; i++) {
            Arrays.fill(cards_players[i], 0);
            Arrays.fill(tracks_ownership[i], false);
            Arrays.fill(tickets[i], false);
            Arrays.fill(boats_locos_buildt[i], 0);
        }

        init_allocation();
    }

    private void init_allocation() {
        for (int i = 0; i != 28; i++) {
            cards_players[i % 4][deck.boats.pop()]++;
        }
        for (int i = 0; i != 12; i++) {
            cards_players[i % 4][deck.locos.pop()]++;
        }
        for (int i = 0; i != 3; i++) {
            lane[i] = deck.locos.pop();
        }
        for (int j = 3; j != 6; j++) {
            lane[j] = deck.boats.pop();
        }
    }
    private void build_track(connection track, int color){
        tracks_ownership[current_player_move][track.id]=true;
        if (track.boat){
            int d=((track.building_cost )/2)-cards_players[current_player_move][color];
            boats_locos_buildt[current_player_move][color]=
            cards_players[current_player_move][color]=cards_players[current_player_move][color]-d;
            while (cards_players[current_player_move][color+7]!=0 && d!=0){
                d--;
                cards_players[current_player_move][color+7]--;
            }
            if(d!=0){
                while (cards_players[current_player_move][23]!=0 && d!=0){
                    d--;
                    cards_players[current_player_move][23]--;
                }//TODO !!!!!
            }

        }
    }


}
