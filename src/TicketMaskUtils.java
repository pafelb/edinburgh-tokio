import java.util.ArrayList;
import java.util.List;

public final class TicketMaskUtils {

    private TicketMaskUtils() {}

    // Precompute mask tables for 5-ticket start (keep >=3) and 4-ticket later (keep >=1)
    // Each mask is a boolean[] of length numTickets (5 or 4)
    // TODO: legal mask for exploration phase to only choose min amount of tickets

    public static final List<boolean[]> START_MASKS_5_FROM_5_MIN3 =
            generateMasks(5, 3);   // size will be 16

    public static final List<boolean[]> MID_MASKS_4_FROM_4_MIN1 =
            generateMasks(4, 1);   // size will be 15

    public static final int NUM_TICKET_MASKS =
            Math.max(START_MASKS_5_FROM_5_MIN3.size(), MID_MASKS_4_FROM_4_MIN1.size()); // = 16

    /**
     * Generate all bit masks for numTickets where the number of 1-bits >= minKeep.
     * Each mask is boolean[numTickets] with true = keep this ticket.
     */
    public static List<boolean[]> generateMasks(int numTickets, int minKeep) {
        List<boolean[]> result = new ArrayList<>();
        int maxMask = 1 << numTickets;  // 2^numTickets

        for (int mask = 0; mask < maxMask; mask++) {
            int bits = Integer.bitCount(mask);
            if (bits >= minKeep) {
                boolean[] pattern = new boolean[numTickets];
                for (int i = 0; i < numTickets; i++) {
                    pattern[i] = ((mask >> i) & 1) == 1;
                }
                result.add(pattern);
            }
        }

        return result;
    }

    /**
     * Build the legal mask for the ticket-selection head.
     * - For 5 tickets: all START masks are legal → first 16 entries = 1
     * - For 4 tickets: only MID masks are legal → first 15 entries = 1, last one = 0
     */
    public static int[] buildLegalMaskForTicketSelection(int numOfferedTickets) {
        int[] legal = new int[NUM_TICKET_MASKS];

        if (numOfferedTickets == 5) {
            // All 16 patterns from START_MASKS are valid
            for (int i = 0; i < START_MASKS_5_FROM_5_MIN3.size(); i++) {
                legal[i] = 1;
            }
            // Any remaining indices (if any) stay 0
        } else if (numOfferedTickets == 4) {
            // 15 valid patterns for mid-game; last one illegal
            for (int i = 0; i < MID_MASKS_4_FROM_4_MIN1.size(); i++) {
                legal[i] = 1;
            }
            // any other entries remain 0
        } else {
            // For now: no valid ticket selection for other counts
            // (you could extend this if rules change)
        }

        return legal;
    }

    /**
     * Decode a chosen mask index into which of the offered tickets to keep.
     * offeredTickets.size() must be 5 (start) or 4 (later).
     */
    public static <T> List<T> decodeMask(int maskIndex, List<T> offeredTickets) {
        int n = offeredTickets.size();

        boolean[] pattern;
        if (n == 5) {
            pattern = START_MASKS_5_FROM_5_MIN3.get(maskIndex);
        } else if (n == 4) {
            pattern = MID_MASKS_4_FROM_4_MIN1.get(maskIndex);
        } else {
            throw new IllegalArgumentException("Unsupported number of offered tickets: " + n);
        }

        List<T> selected = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (pattern[i]) {
                selected.add(offeredTickets.get(i));
            }
        }
        return selected;
    }
}
