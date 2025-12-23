import java.util.Collections;
import java.util.Random;
import java.util.Stack;

public class deck {

    public deck() {
        init();
    }


    @Override
    public String toString() {
        return "deck{" +
                "boats=" + boats.toString() +
                ", locos=" + locos.toString() +
                '}';
    }

    /*
        0 white s
        1 green s
        2 red s
        3 black s
        4 pink s
        5 yellow s
        6 white d
        7 green d
        8 red d
        9 black d
        10 pink d
        11 yellow d
        12 white l
        13 green l
        14 red l
        15 black l
        16 pink l
        17 yellow l
        18 white h
        19 green h
        20 red h
        21 black h
        22 pink h
        23 yellow h
        24 joker

           */

    public  String cardIdToString(int id) {
        // Lookup table of size 25 (0..24)
        final String[] names = {
                "white s",   // 0
                "green s",   // 1
                "red s",     // 2
                "black s",   // 3
                "pink s",    // 4
                "yellow s",  // 5

                "white d",   // 6
                "green d",   // 7
                "red d",     // 8
                "black d",   // 9
                "pink d",    // 10
                "yellow d",  // 11

                "white l",   // 12
                "green l",   // 13
                "red l",     // 14
                "black l",   // 15
                "pink l",    // 16
                "yellow l",  // 17

                "white h",   // 18
                "green h",   // 19
                "red h",     // 20
                "black h",   // 21
                "pink h",    // 22
                "yellow h",  // 23

                "joker"      // 24
        };

        if (id < 0 || id >= names.length) {
            throw new IllegalArgumentException("Invalid card id: " + id);
        }

        return names[id];
    }

    public Stack<Integer> boats = new Stack<>();
    public Stack<Integer> locos = new Stack<>();
    public Stack<Integer> boats_discard = new Stack<>();
    public Stack<Integer> locos_discard = new Stack<>();
    public Stack<ticket> tickets = new Stack<>();
    public Stack<ticket> discardTickets = new Stack<>();

    public final int[] ammountofCards = {4,4,4,4,4,4,6,6,6,6,6,6,7,7,7,7,7,7,4,4,4,4,4,4,14};

    private void init() {
        for (int i = 0; i <= 11; i++) {
            for (int j = 0; j != ammountofCards[i]; j++)
                boats.push(i);
        }
        for (int i = 0; i != 14; i++) {
            locos.push(24);
        }
        for (int i = 12; i <= 23; i++) {
            for (int j = 0; j != ammountofCards[i]; j++)
                locos.push(i);

        }
        Collections.shuffle(locos, new Random(0));
        Collections.shuffle(boats, new Random(0));
    }
    public void shuffle(){
        if (boats.isEmpty()){
           boats = (Stack<Integer>) boats_discard.clone();
           Collections.shuffle(boats);
           boats_discard.clear();
        }
        if (locos.isEmpty()){
            locos = (Stack<Integer>) locos_discard.clone();
            Collections.shuffle(locos);
            locos_discard.clear();
        }
    }


}
