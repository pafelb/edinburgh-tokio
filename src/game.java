import org.datavec.api.writable.Text;

import java.util.*;

public class game {
    private deck deck;
    private int remaining_rounds = 3;
    private int[][] cards_players = new int[4][25];
    private boolean[][] tracks_ownership = new boolean[4][129];
    private boolean[][] tickets = new boolean[4][62];
    private int[][] boats_locos_buildt = new int[4][4];
    private int[] lane = new int[6];
    private int pointsPunished = 0;
    private gamestate gamestate;
    private int[] splitLocos = new int[4];

    private int current_player_move = 0;
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


    //TODO DOUBLE BOAT SAFE STRATEGY NOT IMPLEMENTED; ADD LATER
    //TODO CLEAN UP COLOR PREFERENCE; EITHER ALL HERE OR NONE
    private void build_track(connection track, int color) {
        float[] temp = agent.getCachedColorPreferenceProbsFromCache(null);
        int need = track.building_cost;
        Integer[] indices = new Integer[temp.length];
        for (int i = 0; i < temp.length; i++) {
            indices[i] = i;
        }
        Arrays.sort(indices, Comparator.comparingDouble(i -> temp[i]));

        if (track.boat) {

            // Optional safety check (cheap) – helps catch mask bugs
            int singles = cards_players[current_player_move][color];
            int doubles = cards_players[current_player_move][color + 6];
            int jokers = cards_players[current_player_move][24];
            int maxCoverage = singles + jokers + 2 * doubles;
            if (maxCoverage < need) {
                throw new IllegalStateException("Illegal build (mask bug): not enough boat cards for track " + track.id);
            }

            // Pay: doubles first (2 coverage each), then singles (1), then jokers (1)
            while (need > 1 && cards_players[current_player_move][color + 6] > 0) {
                cards_players[current_player_move][color + 6]--;
                deck.discardAfterTrackBuild(color + 6);
                need -= 2;
            }
            while (need > 0 && cards_players[current_player_move][color] > 0) {
                cards_players[current_player_move][color]--;
                deck.discardAfterTrackBuild(color);
                need -= 1;
            }
            while (need > 0 && cards_players[current_player_move][24] > 0) {
                cards_players[current_player_move][24]--;
                deck.discardAfterTrackBuild((24));
                need -= 1;
            }

            // should always be 0 if mask was correct
            if (need != 0) {
                throw new IllegalStateException("Payment failed unexpectedly for track " + track.id);
            }

            // Now claim ownership (after successful payment)
            tracks_ownership[current_player_move][track.id] = true;

        } else {
            if (track.segmented) {
                outer:
                while (need > 0) {

                    // 1) two normal train cards
                    for (int k = 0; k < 6; k++) {
                        int c = indices[k];
                        if (cards_players[current_player_move][c + 12] > 1) {
                            need -= 2;
                            cards_players[current_player_move][c + 12] -= 2;
                            deck.discardAfterTrackBuild(c + 12, 2);
                            continue outer;
                        }
                    }

                    // 2) one normal + one harbor train card (same color)
                    for (int k = 0; k < 6; k++) {
                        int c = indices[k];
                        if (cards_players[current_player_move][c + 12] > 0
                                && cards_players[current_player_move][c + 18] > 0) {
                            need -= 2;
                            cards_players[current_player_move][c + 12]--;
                            cards_players[current_player_move][c + 18]--;
                            deck.discardAfterTrackBuild(c + 12);
                            deck.discardAfterTrackBuild(c + 18);
                            continue outer;
                        }
                    }

                    // 3) two harbor train cards
                    for (int k = 0; k < 6; k++) {
                        int c = indices[k];
                        if (cards_players[current_player_move][c + 18] > 1) {
                            need -= 2;
                            cards_players[current_player_move][c + 18] -= 2;
                            deck.discardAfterTrackBuild(c + 18, 2);
                            continue outer;
                        }
                    }

                    // 4) one normal + joker
                    for (int k = 0; k < 6; k++) {
                        int c = indices[k];
                        if (cards_players[current_player_move][c + 12] > 0
                                && cards_players[current_player_move][24] > 0) {
                            need -= 2;
                            cards_players[current_player_move][c + 12]--;
                            cards_players[current_player_move][24]--;
                            deck.discardAfterTrackBuild(c + 12);
                            deck.discardAfterTrackBuild(24);
                            continue outer;
                        }
                    }

                    // 5) one harbor + joker
                    for (int k = 0; k < 6; k++) {
                        int c = indices[k];
                        if (cards_players[current_player_move][c + 18] > 0
                                && cards_players[current_player_move][24] > 0) {
                            need -= 2;
                            cards_players[current_player_move][c + 18]--;
                            cards_players[current_player_move][24]--;
                            deck.discardAfterTrackBuild(c + 18);
                            deck.discardAfterTrackBuild(24);
                            continue outer;
                        }
                    }

                    // 6) two jokers
                    if (cards_players[current_player_move][24] > 1) {
                        need -= 2;
                        cards_players[current_player_move][24] -= 2;
                        deck.discardAfterTrackBuild(24, 2);
                        continue;
                    }

                    // If we got here, the legal mask was wrong or state is inconsistent
                    throw new IllegalStateException("Segmented payment failed unexpectedly for track " + track.id);
                }

                tracks_ownership[current_player_move][track.id] = true;
            } else {
                while (cards_players[current_player_move][color + 12] != 0 && need != 0) {
                    cards_players[current_player_move][color + 12]--;
                    need--;
                }
                while (cards_players[current_player_move][color + 18] != 0 && need != 0) {
                    cards_players[current_player_move][color + 18]--;
                    need--;
                }
                while (cards_players[current_player_move][24] != 0 && need != 0) {
                    cards_players[current_player_move][24]--;
                    need--;
                }

                if (need != 0) {
                    throw new IllegalStateException("Payment failed unexpectedly for track " + track.id);
                }

                // Now claim ownership (after successful payment)
                tracks_ownership[current_player_move][track.id] = true;

            }


        }
    }

    /**
     * Build the state vector for the "ticket selection" phase, given the offered tickets.
     * You implement this using your existing state encoder (set offer one-hots, set phase flag, etc.).
     */
    @FunctionalInterface
    public interface TicketOfferStateBuilder {
        float[] buildState(List<ticket> offeredTickets, boolean isStartOfGame);
    }

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
    public TicketDrawResult drawDestinationTickets(
            boolean isStartOfGame,
            Deque<ticket> destinationDeck,
            Deque<Integer> destinationDiscard,
            TicketToRideAgent agent,
            TicketOfferStateBuilder stateBuilder //TODO DELETE ALL AGENT REFERENCES, CLASS CONTAINS AGENT
    ) {
        int drawCount = isStartOfGame ? 5 : 4;

        // 1) Draw offered tickets
        List<ticket> offered = new ArrayList<>(drawCount);
        for (int i = 0; i < drawCount; i++) {
            if (destinationDeck.isEmpty()) {
                throw new IllegalStateException("Destination deck is empty");
            }
            offered.add(destinationDeck.removeFirst());
        }

        // 2) Build state for ticket selection phase (you control encoding)
        float[] state = stateBuilder.buildState(offered, isStartOfGame);  //TODO STATEBUILDER

        // 3) Get probabilities from the ticket-selection head (size = TicketMaskUtils.NUM_TICKET_MASKS = 16)
        float[] probs = agent.getTicketMaskProbabilities(state);

        // 4) Legal mask from your utils (int[16] with 1 = legal, 0 = illegal)
        int[] legalInt = TicketMaskUtils.buildLegalMaskForTicketSelection(offered.size());

        // 5) Apply mask + renormalize (same behavior as your agent.applyMask, but that method is private)
        float sum = 0f;
        for (int i = 0; i < probs.length; i++) {
            if (i >= legalInt.length || legalInt[i] == 0) {
                probs[i] = 0f;
            } else {
                sum += probs[i];
            }
        }
        if (sum > 0f) {
            for (int i = 0; i < probs.length; i++) {
                probs[i] /= sum;
            }
        } else {
            // fallback: if everything became 0 (shouldn't happen), pick first legal mask
            for (int i = 0; i < legalInt.length; i++) {
                if (legalInt[i] == 1) {
                    probs[i] = 1f;
                    break;
                }
            }
        }

        // 6) Choose mask index (sampling for exploration; you can replace with argmax if you want deterministic)
        int chosenMaskIndex = agent.sampleAction(probs);

        // 7) Decode kept tickets using your utils (this already handles 5-ticket vs 4-ticket internally)
        List<ticket> kept = TicketMaskUtils.decodeMask(chosenMaskIndex, offered);

        // 8) Everything else is discarded
        List<ticket> discarded = new ArrayList<>(offered);
        discarded.removeAll(kept);

        // 9) Put discarded tickets into discard pile (optional; you said you'll interpret further later)
        // If you don't want to discard here yet, just remove this loop.
        deck.discardShuffle(discarded);

        return new TicketDrawResult(kept, discarded);
    }

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
        gamestate.update(offered);

        // 3) AGENT EVALUATE ON NEW STATE
        agent.evaluate(gamestate.toVector());

        // 4) BUILD LEGAL MASK (int[16] -> boolean[16])
        int[] legalInt = TicketMaskUtils.buildLegalMaskForTicketSelection(offered.size());
        boolean[] legalBools = new boolean[TicketMaskUtils.NUM_TICKET_MASKS];

        for (int i = 0; i < legalBools.length; i++) {
            legalBools[i] = (i < legalInt.length && legalInt[i] == 1);
        }

        // 5) GET PROBS WITH MASK + CHOOSE MASK INDEX
        float[] probs = agent.getTicketMaskProbabilitiesFromCache(legalBools);
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
    private void changeBoatLocoRatio(int newLocoAmount, boolean punish){

    }



}
