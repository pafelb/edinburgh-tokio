import java.util.*;

public class game {
    private deck deck;
    private int remaining_rounds = 3;
    private int[][] cards_players = new int[4][25];
    private boolean[][] tracks_ownership = new boolean[4][129];
    private boolean[][] tickets = new boolean[4][62];
    private int[][] boats_locos_buildt = new int[4][2];
    private int[] lane = new int[6];
    private int pointsPunished = 0;
    private gamestate gamestate;
    private int[] splitLocos = new int[4];
    private map map;
    public boolean firstRound = true;
    private float[][] destinationFullfillment = new float[4][65]; //TODO EFFICIENT CALCUlATION OF FULLFILLMENT see gamestate for info
    private int current_player_move = 0;
    TicketToRideAgent agent = new TicketToRideAgent();
    private int[][] knownCards = new int[4][25];
    private int[][] unknownCards = new int[4][2];
    private int discardBoats;
    private int discardLokos;
    private int randomBoats;
    private int randomLokos;
    private int[]discardStack;

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
        while (remaining_rounds != 0) {


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

    public static class TicketDrawResult {
        public final List<ticket> kept;
        public final List<ticket> discarded;

        public TicketDrawResult(List<ticket> kept, List<ticket> discarded) {
            this.kept = kept;
            this.discarded = discarded;
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

    private void changeSplit(int i) {
        splitLocos[current_player_move] = agent.sampleAction(agent.getCachedFleetCompositionProbs(maskBuilderSplit(new float[i])));
    }

    //TODO !!!!!!! CHECK FOR UPDATE AFTER BUILDING DOUBLE BOAT CARD FOR BOATS LOCOS BUILDT
    private boolean[] maskBuilderSplit(float[] r) {
        boolean[] mask = new boolean[r.length];
        Arrays.fill(mask, true);
        int boats = boats_locos_buildt[current_player_move][0] - 35;
        int locos = boats_locos_buildt[current_player_move][1] - 10;
        while (locos > 0) {
            mask[locos] = false;
            locos--;
        }
        while (boats > 0) {
            mask[boats] = false;
            boats--;
        }
        mask[splitLocos[current_player_move]] = r[1337 /*TODO TAKE VECTOR SPACE*/] != 0.0f; // IF FIRSTROUND IGNORE THAT YOU CANNOT take your current split
        return mask;
    }

    //TODO UPDATE TO MAKE IT UPDATE GAMESTATE
    private float[] buildStateVectorWithTicketOffer(int[] i, int j) {
        return new float[agent.STATE_SIZE];
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

    private void buildProperTrack() {
        int track = agent.sampleAction(agent.getCachedTrackSelectionProbs(maskTrackBuildingOptions()));
        List<Integer> IdsToBuild = findBestPayment(map.getConnections()[track], agent.getCachedColorPreferenceProbs(null));
        if (IdsToBuild != null) {
            tracks_ownership[current_player_move][track] = true;
            for (Integer i : IdsToBuild) {
                cards_players[current_player_move][i]--;
            }
        } else {
            //OH OH only happens if legal mask not working
        }
    }

    /**
     * Finds the legal card combination with the lowest total AI preference cost.
     * Updated: building_cost now represents the total units (e.g., 6 for a 3-unit segmented track).
     */
    public List<Integer> findBestPayment(connection t, float[] aiPrefs) {
        if (aiPrefs == null) return null; // Safety check for unitialized agent

        if (t.segmented) {
            return findBestSegmentedPayment(t, aiPrefs);
        } else {
            return findBestStandardPayment(t, aiPrefs);
        }
    }

    private List<Integer> findBestSegmentedPayment(connection t, float[] aiPrefs) {
        List<Integer> bestCombination = null;
        float minTotalCost = Float.MAX_VALUE;

        // Iterate through all 6 colors (0:white to 5:yellow)
        for (int colorIdx = 0; colorIdx < 6; colorIdx++) {
            // Segmented tracks use Train Singles (12-17), Harbor Trains (18-23), and Jokers (24)
            List<Integer> candidates = new ArrayList<>();
            int trainSingleId = colorIdx + 12;
            int harborTrainId = colorIdx + 18;

            for (int i = 0; i < cards_players[current_player_move][trainSingleId]; i++) candidates.add(trainSingleId);
            for (int i = 0; i < cards_players[current_player_move][harborTrainId]; i++) candidates.add(harborTrainId);
            for (int i = 0; i < cards_players[current_player_move][24]; i++) candidates.add(24);

            // Per your correction: a segmented track of length 3 has building_cost = 6
            if (candidates.size() < t.building_cost) continue;

            // Sort by AI preference to find the "cheapest" non-human strategic choice
            candidates.sort(Comparator.comparingDouble(id -> aiPrefs[id]));

            List<Integer> currentCombination = new ArrayList<>(candidates.subList(0, t.building_cost));
            float currentCost = 0;
            for (int id : currentCombination) currentCost += aiPrefs[id];

            if (currentCost < minTotalCost) {
                minTotalCost = currentCost;
                bestCombination = currentCombination;
            }
        }
        return bestCombination;
    }

    private List<Integer> findBestStandardPayment(connection t, float[] aiPrefs) {
        List<Integer> bestCombination = null;
        float minTotalCost = Float.MAX_VALUE;

        // Handle Gray tracks (ID 7) vs Colored tracks
        List<Integer> validColors = new ArrayList<>();
        if (t.color == 7) {
            for (int i = 0; i < 6; i++) validColors.add(i);
        } else {
            validColors.add(t.color);
        }

        for (int colorIdx : validColors) {
            List<Integer> candidates = new ArrayList<>();
            if (t.boat) {
                // Boat cards: singles (0-5) and doubles (6-11)
                for (int i = 0; i < cards_players[current_player_move][colorIdx]; i++) candidates.add(colorIdx);
                for (int i = 0; i < cards_players[current_player_move][colorIdx + 6]; i++) candidates.add(colorIdx + 6);
            } else {
                // Train cards: singles (12-17) and harbor trains (18-23)
                for (int i = 0; i < cards_players[current_player_move][colorIdx + 12]; i++)
                    candidates.add(colorIdx + 12);
                for (int i = 0; i < cards_players[current_player_move][colorIdx + 18]; i++)
                    candidates.add(colorIdx + 18);
            }
            for (int i = 0; i < cards_players[current_player_move][24]; i++) candidates.add(24);

            List<Integer> currentCombination = solveStandardMinCost(candidates, t.building_cost, aiPrefs);

            if (currentCombination != null) {
                float currentCost = 0;
                for (int id : currentCombination) currentCost += aiPrefs[id];
                if (currentCost < minTotalCost) {
                    minTotalCost = currentCost;
                    bestCombination = currentCombination;
                }
            }
        }
        return bestCombination;
    }

    private List<Integer> solveStandardMinCost(List<Integer> candidates, int requiredCost, float[] aiPrefs) {
        candidates.sort(Comparator.comparingDouble(id -> aiPrefs[id]));

        List<Integer> chosenCards = new ArrayList<>();
        int currentCoverage = 0;

        for (int cardId : candidates) {
            if (currentCoverage >= requiredCost) break;

            // IDs 6-11 are Double Boats, worth 2 units
            int coverage = (cardId >= 6 && cardId <= 11) ? 2 : 1;

            chosenCards.add(cardId);
            currentCoverage += coverage;
        }

        return (currentCoverage >= requiredCost) ? chosenCards : null;
    }


    private boolean[] maskTrackBuildingOptions() {
        boolean[] mask = new boolean[map.getConnections().length];
        int[] max = new int[15];
        int wBoat, gBoat, rBoat, bBoat, pBoat, yBoat, wLoco, gLoco, rLoco, bLoco, pLoco, yLoco, segmented = 0;
        for (int i = 0; i != 6; i++) {
            max[i] = cards_players[current_player_move][i] * 2 + cards_players[current_player_move][i + 6] + cards_players[current_player_move][24];
        }
        for (int i = 6; i != 14; i++) {
            max[i] = cards_players[current_player_move][i + 12] + cards_players[current_player_move][i + 18] + cards_players[current_player_move][24];
        }
        //TODO IMPLEMENT SEGMENTED
        //first get already doubled cards
        int d = 0;
        int t = 0;
        for (int i = 0; i != 12; i++) {
            d += (cards_players[current_player_move][i + 12] + cards_players[current_player_move][i + 18]) / 2;
            t += (cards_players[current_player_move][i + 12] + cards_players[current_player_move][i + 18]) % 2;
        }
        max[12] = d + Math.min(Math.min(t, cards_players[current_player_move][24]), cards_players[current_player_move][24] / 2);
        max[13] = Arrays.stream(max, 0, 6).max().getAsInt();
        max[14] = Arrays.stream(max, 6, 12).max().getAsInt();
        for (connection c : map.getConnections()) {
            boolean b = c.boat;
            int r = !b ? 1 : 0;
            if (c.color == 7) {
                if (max[13 + r] >= c.building_cost)
                    mask[c.id] = true;

            } else {
                if (max[c.color + r * 6] >= c.building_cost)
                    mask[c.id] = true;
            }


        }
        return mask;
    }


    //TODO FUNCTION FOR PREFERENCE MAXIMIASTION


    /**
     * Full destination-ticket flow:
     * - draw 5 (start) or 4 (later) tickets from destinationDeck
     * - build the legal mask using TicketMaskUtils
     * - call agent.getTicketMaskProbabilities(state)
     * - apply legal mask (renormalize)
     * - pick a mask index (sample)
     * - decode kept tickets using TicketMaskUtils.decodeMask(...)
     * - return kept + discarded lists
     * <p>
     * This function does NOT update your player ticket storage yet (you said you'll interpret/apply later).
     */


    private void drawTickets(boolean firstround) {
        Stack<ticket> offered = new Stack<>();
        List<ticket> toDiscard;

        // 1) DRAW TICKETS FROM DECK
        int drawCount = firstround ? 5 : 4;

        for (int i = 0; i < drawCount; i++) {
            if (deck.tickets.isEmpty()) break;
            offered.push(deck.tickets.pop());
        }
        offered.sort(Comparator.comparingInt(t -> t.id));

        // 2) UPDATE GAMESTATE WITH OFFER
        gamestate.update();

        // 3) AGENT EVALUATE ON NEW STATE
        agent.evaluate(gamestate.toVector());

        // 4) BUILD LEGAL MASK (int[16] -> boolean[16])
        int[] legalInt = TicketMaskUtils.buildLegalMaskForTicketSelection(offered.size());
        boolean[] legalBools = new boolean[TicketMaskUtils.NUM_TICKET_MASKS];

        for (int i = 0; i < legalBools.length; i++) {
            legalBools[i] = (i < legalInt.length && legalInt[i] == 1);
        }

        // 5) GET PROBS WITH MASK + CHOOSE MASK INDEX
        float[] probs = agent.getCachedTicketSelectionProbs(legalBools);
        int maskIndex = agent.sampleAction(probs);

        if (maskIndex < 0 || maskIndex >= legalBools.length || !legalBools[maskIndex]) {
            throw new IllegalStateException("Agent chose illegal ticket mask: " + maskIndex);
        }

        // 6) DECODE SELECTION
        // decodeMask likely expects a List; Stack is fine but to be explicit:
        List<ticket> offeredList = new ArrayList<>(offered);
        List<ticket> keep = TicketMaskUtils.decodeMask(maskIndex, offeredList);

        // 7) DISCARD THE REST
        toDiscard = new ArrayList<>(offeredList);
        toDiscard.removeAll(keep);
        deck.discardShuffle(toDiscard);

        // 8) ADD KEPT TICKETS TO PLAYER
        for (ticket t : keep) {
            this.tickets[current_player_move][t.id] = true;
        }
    }

    private void changeBoatLocoRatio(int newLocoAmount, boolean punish) {

    }


    public deck getDeck() {
        return deck;
    }

    public int getRemaining_rounds() {
        return remaining_rounds;
    }

    public int[][] getCards_players() {
        return cards_players;
    }

    public boolean[][] getTracks_ownership() {
        return tracks_ownership;
    }

    public boolean[][] getTickets() {
        return tickets;
    }

    public int[][] getBoats_locos_buildt() {
        return boats_locos_buildt;
    }

    public int[] getLane() {
        return lane;
    }

    public int getPointsPunished() {
        return pointsPunished;
    }

    public gamestate getGamestate() {
        return gamestate;
    }

    public int[] getSplitLocos() {
        return splitLocos;
    }

    public map getMap() {
        return map;
    }

    public int getCurrent_player_move() {
        return current_player_move;
    }

    public float[][] getDestinationFullfillment() {
        return destinationFullfillment;
    }

    public boolean[][] getHarbors() {
        return harbors;
    }

    private boolean[][] harbors = new boolean[4][37]; //TODO IN SETUP

    public int[][] getKnownCards() {
        return knownCards;
    }

    public int[][] getUnknownCards() {
        return unknownCards;
    }

    public int getRandomLokos() {
        return randomLokos;
    }

    public int getRandomBoats() {
        return randomBoats;
    }

    public int getDiscardBoats() {
        return discardBoats;
    }

    public int getDiscardLokos() {
        return discardLokos;
    }

    public int[] getDiscardStack() {
        return discardStack;
    }
}
