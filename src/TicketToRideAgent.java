import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Deep Learning Agent for Ticket to Ride: Rails & Sails using Deeplearning4j
 * Features hierarchical action spaces with shared backbone
 */
public class TicketToRideAgent {


    // ===== ADJUST THESE IF YOUR STATE ENCODING CHANGES =====
    // ===== GAME/STATE CONSTANTS =====
    public static final int STATE_SIZE = 645; // keep your current value

    private static final int NUM_ROUTES = 129;
    private static final int NUM_FACE_UP_CARDS = 6;

    // Ticket masks: start game uses 16 legal masks for "keep >=3 of 5"
// mid-game uses 15 legal masks for "keep >=1 of 4" (you mask illegal ones)
    private static final int NUM_TICKET_MASKS = 16;

    // Draw options: 6 face-up + 2 blind draws (boat deck + loco/train deck)
    private static final int NUM_DRAW_OPTIONS = NUM_FACE_UP_CARDS + 2; // 8

    // Main action head: ONLY the 5 action types
    private static final int NUM_MAIN_ACTIONS = 5;

    private static final int NUM_FLEET_OPTIONS = 14;
    private static final int NUM_BOAT_MODES = 5;

    // Calculated action space sizes (don't change these)
    private static final int NUM_DRAW_ACTIONS = 2 + NUM_FACE_UP_CARDS; // 2 decks + 6 face-up = 8


    // Action type boundaries for main action head

    public static final int CLAIM_ROUTE_START = 0;  // Start of route claiming
    public static final int CLAIM_ROUTE_END = 128;   // End of route claiming (8 + 80 - 1)
    public static final int DRAW_DESTINATIONS = 129; // Draw destination tickets
    public static final int DRAW_TRAIN_DECK = 130;    // Draw from train deck
    public static final int DRAW_BOAT_DECK = 131;     // Draw from boat deck
    public static final int DRAW_FACEUP_START = 132;  // Start of face-up cards (indices 2-7)
    public static final int DRAW_FACEUP_END = 138;    // End of face-up cards
    public static final int BUILD_HARBOUR_START = 139;
    public static final int BUILD_HARBOUR_END = 176;
    public static final int REBALANCE_FLEET = 177;//

    // =======================
    // CACHED FORWARD PASS API
    // =======================

    // Cache: last encoded features and head outputs for the most recent evaluate() call
    private INDArray cachedFeatures = null;

    private float[] cachedMainActionProbs = null;
    private float[] cachedTakeMaterialProbs = null;
    private float[] cachedRefillDeckProbs = null;
    private float[] cachedTicketSelectionProbs = null;
    private float[] cachedFleetCompositionProbs = null;
    private float[] cachedColorPreferenceProbs = null;
    private float[] cachedRessourceValueProbs = null;
    private float[] cachedDoubleBiasProbs = null;
    private float[] cachedTrackSelectionProbs = null;


    // ===== MAIN ACTION ENUM (optional but recommended) =====
    public enum MainAction {
        DRAW_CARDS,      // 0
        CLAIM_ROUTE,     // 1
        DRAW_TICKETS,    // 2
        BUILD_HARBOR,  //TODO  // 3
        EXCHANGE_PIECES  // 4
    }

    //NETWORK ENCODER
    private MultiLayerNetwork sharedEncoder;
    //NETWORK HEADS
    private MultiLayerNetwork mainActionHead;       //MAIN ACTIONS
    private MultiLayerNetwork takeMaterialHead;     //DECIDES WHAT CARD TO TAKE FROM DECK
    private MultiLayerNetwork refillDeckHead;//DECIDES WETHER LAY BOAT OR LOCO OPEN
    private MultiLayerNetwork ticketSelectionHead;  //DECIDES WHICH TICKET COMBINATION TO TAKE
    private MultiLayerNetwork fleetCompositionHead; //DECIDES WHAT COMPOSITION IS BEST
    private MultiLayerNetwork colorPreferenceHead;  //GIVES PREFERENCES OF COLORS
    private MultiLayerNetwork resourceValueHead;    //GIVES VALUE FOR 3 POSS OF CARDS WHEN BUILDING THINGS
    private MultiLayerNetwork doubleBiasHead;       //GIVES VALUE OF KEEPING DOUBLE BOAT IN SPEC COLOR
    private MultiLayerNetwork trackSelectionHead;   //GIVES ID OF TRACK TO BUILDT


    private Random random = new Random();

    public TicketToRideAgent() {
        initializeNetworks();
    }

    private void initializeNetworks() {
        // ===== SHARED ENCODER - Processes game state into features =====
        // This network learns to understand the game state
        // Architecture: STATE_SIZE -> 256 -> 128 -> 128

        MultiLayerConfiguration encoderConf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))  // Learning rate: 0.001 (can tune this)
                .list()
                // Layer 1: Extract basic patterns from raw state
                .layer(new DenseLayer.Builder()
                        .nIn(STATE_SIZE)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                // Layer 2: Combine patterns into higher-level features
                .layer(new DenseLayer.Builder()
                        .nIn(256)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                // Layer 3: Refine understanding
                .layer(new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .build();

        sharedEncoder = new MultiLayerNetwork(encoderConf);
        sharedEncoder.init();

        // ===== MAIN ACTION HEAD - Decides what to do this turn =====
        // Outputs probabilities for: draw cards, claim route, draw destinations, or rebalance

        MultiLayerConfiguration mainConf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(128)
                        .nOut(NUM_MAIN_ACTIONS)  // 90 actions total
                        .activation(Activation.SOFTMAX)  // Converts to probabilities
                        .build())
                .build();

        mainActionHead = new MultiLayerNetwork(mainConf);
        mainActionHead.init();
        mainActionHead.setListeners(new ScoreIterationListener(100));

        // ===== FLEET COMPOSITION HEAD - Decides locomotive/boat split =====
        // Only used when rebalancing fleet (costs 1 turn)
        // Outputs probabilities for 10-30 locomotives (rest are boats)

        MultiLayerConfiguration fleetConf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(64)
                        .nOut(NUM_FLEET_OPTIONS)  // 21 options (10 to 30 locomotives)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        fleetCompositionHead = new MultiLayerNetwork(fleetConf);
        fleetCompositionHead.init();

        // ===== SECOND CARD HEAD - Decides which card to draw as 2nd card =====
        // Only used after first card draw (if allowed to draw second card)
        // Same options as first draw: 2 decks + 6 face-up cards = 8 options

        MultiLayerConfiguration secondCardConf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(64)
                        .nOut(NUM_DRAW_ACTIONS)  // 8 options: 2 decks + 6 face-up
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        takeMaterialHead = new MultiLayerNetwork(secondCardConf);
        takeMaterialHead.init();


        // ===== TICKET SELECTION HEAD =====
        // Input: shared encoder features (e.g. 128)
        // Output: 16 mask options (some illegal depending on context)
        MultiLayerConfiguration ticketHeadConf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(128)       // same as shared encoder output size
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(64)
                        .nOut(NUM_TICKET_MASKS)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        ticketSelectionHead = new MultiLayerNetwork(ticketHeadConf);
        ticketSelectionHead.init();

        //Choose boat or loko after drawing card head
        MultiLayerConfiguration refillDeckHeadConf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(64)
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();
        refillDeckHead = new MultiLayerNetwork(refillDeckHeadConf);
        refillDeckHead.init();

        // COLOR PREF HEAD
        MultiLayerConfiguration colorPreferenceConf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(64)
                        .nOut(6)
                        .activation(Activation.SIGMOID)
                        .build())
                .build();
        colorPreferenceHead = new MultiLayerNetwork(colorPreferenceConf);
        colorPreferenceHead.init();

        //decides opportunity cost of each double single joker/normal harbour wildcard/harbour single wildcard
        MultiLayerConfiguration resourceValueConf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(32)
                        .nOut(3)                 // double, single, wild
                        .activation(Activation.SIGMOID)
                        .build())
                .build();
        resourceValueHead = new MultiLayerNetwork(resourceValueConf);
        resourceValueHead.init();

        //gives bias for choosing to build double boats instead of single boats
        MultiLayerConfiguration doubleBiasConf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(32)
                        .nOut(6)                 // per-color double bias
                        .activation(Activation.SIGMOID)
                        .build())
                .build();
        doubleBiasHead = new MultiLayerNetwork(doubleBiasConf);
        doubleBiasHead.init();

        MultiLayerConfiguration trackSelectionConf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(128)
                        .nOut(128)//TODO TRACK AMMOUNT
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();
        trackSelectionHead = new MultiLayerNetwork(trackSelectionConf);
        trackSelectionHead.init();


    }


    /**
     * Sample action from probability distribution (stochastic - for training/exploration)
     * Higher probability actions are chosen more often, but lower probability actions can happen
     *
     * @param probabilities Probability distribution (must sum to ~1.0)
     * @return Index of sampled action
     */
    public int sampleAction(float[] probabilities) {
        float sum = 0;
        for (float p : probabilities) sum += p;

        float rand = random.nextFloat() * sum;
        float cumSum = 0;

        for (int i = 0; i < probabilities.length; i++) {
            cumSum += probabilities[i];
            if (rand <= cumSum) {
                return i;
            }
        }

        return probabilities.length - 1;
    }


    /**
     * Get best action (greedy - for evaluation, not training)
     * Always picks the action with highest probability
     *
     * @param probabilities Probability distribution
     * @return Index of action with highest probability
     */
    public int getBestAction(float[] probabilities) {
        int bestIdx = 0;
        float bestProb = probabilities[0];

        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > bestProb) {
                bestProb = probabilities[i];
                bestIdx = i;
            }
        }

        return bestIdx;
    }

    /**
     * Apply legal action mask to probabilities
     * Sets illegal actions to 0 and renormalizes so probabilities sum to 1
     */
    private float[] applyMask(float[] probs, boolean[] mask) {
        float[] masked = new float[probs.length];
        float sum = 0;

        for (int i = 0; i < probs.length; i++) {
            if (mask[i]) {
                masked[i] = probs[i];
                sum += probs[i];
            }
        }

        // Renormalize
        if (sum > 0) {
            for (int i = 0; i < masked.length; i++) {
                masked[i] /= sum;
            }
        }

        return masked;
    }

    /**
     * Train on batch of experiences using policy gradients
     * Reinforces actions that led to winning, weakens actions that led to losing
     *
     * @param experiences List of (state, action, reward) tuples from multiple games
     */
    private void trainOnBatch(List<Experience> experiences) {
        int batchSize = experiences.size();
        INDArray states = Nd4j.zeros(batchSize, STATE_SIZE);
        INDArray mainActions = Nd4j.zeros(batchSize, NUM_MAIN_ACTIONS);
        INDArray rewards = Nd4j.zeros(batchSize, 1);

        // Separate experiences by type
        List<Experience> fleetExperiences = new ArrayList<>();
        List<Experience> secondCardExperiences = new ArrayList<>();
        List<Experience> boatUsageExperiences = new ArrayList<>();  // NEW

        for (int i = 0; i < batchSize; i++) {
            Experience exp = experiences.get(i);

            // Main head data
            states.putRow(i, Nd4j.create(exp.state));
            mainActions.put(i, exp.mainAction, 1.0);
            rewards.put(i, 0, exp.reward);

            // Fleet head samples
            if (exp.mainAction == REBALANCE_FLEET && exp.fleetAction >= 0) {
                fleetExperiences.add(exp);
            }
            // Second-card samples
            else if (exp.secondCardAction >= 0) {
                secondCardExperiences.add(exp);
            }

            // NEW: boat-usage samples only for route-claim actions
            if (exp.boatUsageAction >= 0 &&
                    exp.mainAction >= CLAIM_ROUTE_START &&
                    exp.mainAction <= CLAIM_ROUTE_END) {

                boatUsageExperiences.add(exp);
            }
        }

        // Train shared encoder + main head (always)
        trainMainHead(states, mainActions, rewards);

        if (!fleetExperiences.isEmpty()) {
            trainFleetHead(fleetExperiences);
        }

        if (!secondCardExperiences.isEmpty()) {
            trainSecondCardHead(secondCardExperiences);
        }

        if (!boatUsageExperiences.isEmpty()) {
            trainBoatUsageHead(boatUsageExperiences);   // NEW
        }
    }


    /**
     * Train the main action head and shared encoder
     */
    private void trainMainHead(INDArray states, INDArray actions, INDArray rewards) {
        // Get features from shared encoder
        INDArray features = sharedEncoder.output(states);

        // Weight actions by rewards (policy gradient)
        // Positive reward -> strengthen this action choice
        // Negative reward -> weaken this action choice
        INDArray weightedActions = actions.mul(rewards);

        // Train main head
        DataSet dataSet = new DataSet(features, weightedActions);
        mainActionHead.fit(dataSet);

        // Backprop through shared encoder
        // (Simplified - in full implementation you'd chain gradients properly)
        DataSet encoderData = new DataSet(states, features);
        sharedEncoder.fit(encoderData);
    }


    /**
     * Train the fleet composition head
     */
    private void trainFleetHead(List<Experience> fleetExperiences) {
        int n = fleetExperiences.size();
        INDArray states = Nd4j.zeros(n, STATE_SIZE);
        INDArray actions = Nd4j.zeros(n, NUM_FLEET_OPTIONS);
        INDArray rewards = Nd4j.zeros(n, 1);

        for (int i = 0; i < n; i++) {
            Experience exp = fleetExperiences.get(i);
            states.putRow(i, Nd4j.create(exp.state));
            actions.put(i, exp.fleetAction, 1.0);  // One-hot encoding
            rewards.put(i, 0, exp.reward);
        }

        INDArray features = sharedEncoder.output(states);
        INDArray weightedActions = actions.mul(rewards);

        DataSet dataSet = new DataSet(features, weightedActions);
        fleetCompositionHead.fit(dataSet);
    }

    /**
     * Train the second card drawing head
     */
    private void trainSecondCardHead(List<Experience> cardExperiences) {
        int n = cardExperiences.size();
        INDArray states = Nd4j.zeros(n, STATE_SIZE);
        INDArray actions = Nd4j.zeros(n, NUM_DRAW_ACTIONS);
        INDArray rewards = Nd4j.zeros(n, 1);

        for (int i = 0; i < n; i++) {
            Experience exp = cardExperiences.get(i);
            states.putRow(i, Nd4j.create(exp.state));
            actions.put(i, exp.secondCardAction, 1.0);  // One-hot encoding
            rewards.put(i, 0, exp.reward);
        }

        INDArray features = sharedEncoder.output(states);
        INDArray weightedActions = actions.mul(rewards);

        DataSet dataSet = new DataSet(features, weightedActions);
        takeMaterialHead.fit(dataSet);
    }


    private void trainBoatUsageHead(List<Experience> boatExperiences) {
        int n = boatExperiences.size();
        INDArray states = Nd4j.zeros(n, STATE_SIZE);
        INDArray actions = Nd4j.zeros(n, NUM_BOAT_MODES);
        INDArray rewards = Nd4j.zeros(n, 1);

        for (int i = 0; i < n; i++) {
            Experience exp = boatExperiences.get(i);
            states.putRow(i, Nd4j.create(exp.state));
            actions.put(i, exp.boatUsageAction, 1.0);  // one-hot over boat modes
            rewards.put(i, 0, exp.reward);
        }

        INDArray features = sharedEncoder.output(states);
        INDArray weightedActions = actions.mul(rewards);

        DataSet dataSet = new DataSet(features, weightedActions);
    }


    /**
     * Save all networks to disk
     */
    public void saveModel(String directory) throws IOException {
        File dir = new File(directory);
        if (!dir.exists()) dir.mkdirs();

        sharedEncoder.save(new File(dir, "shared_encoder.zip"));
        mainActionHead.save(new File(dir, "main_head.zip"));
        fleetCompositionHead.save(new File(dir, "fleet_head.zip"));
        takeMaterialHead.save(new File(dir, "second_card_head.zip"));

        System.out.println("Model saved to " + directory);
    }

    /**
     * Load all networks from disk
     */
    public void loadModel(String directory) throws IOException {
        File dir = new File(directory);

        sharedEncoder = MultiLayerNetwork.load(new File(dir, "shared_encoder.zip"), true);
        mainActionHead = MultiLayerNetwork.load(new File(dir, "main_head.zip"), true);
        fleetCompositionHead = MultiLayerNetwork.load(new File(dir, "fleet_head.zip"), true);
        takeMaterialHead = MultiLayerNetwork.load(new File(dir, "second_card_head.zip"), true);

        System.out.println("Model loaded from " + directory);
    }

    /**
     * Experience tuple for training
     * Stores: state, actions taken, and reward received
     */
    public static class Experience {
        public float[] state;          // Game state when action was taken
        public int mainAction;         // Main action chosen
        public int fleetAction;        // Fleet composition (if used)
        public int secondCardAction;   // Second card (if used)
        public int boatUsageAction;    // NEW: boat mode index (0..NUM_BOAT_MODES-1) if claiming a route, -1 otherwise
        public float reward;
        public int ticketMaskAction;  // -1 if not a ticket-selection step
// Reward for this experience

        public Experience(float[] state,
                          int mainAction,
                          int fleetAction,
                          int secondCardAction,
                          int boatUsageAction,
                          float reward,
                          int ticketMaskAction) {
            this.state = state;
            this.mainAction = mainAction;
            this.fleetAction = fleetAction;
            this.secondCardAction = secondCardAction;
            this.boatUsageAction = boatUsageAction;
            this.ticketMaskAction = ticketMaskAction;
            this.reward = reward;
        }
    }


    /**
     * One "agent call": run the shared encoder ONCE and compute all head outputs.
     * Call this once per decision point (per state).
     * <p>
     * After calling evaluate(state), you can read any head output via the getters below
     * without re-running the encoder.
     */
    public void evaluate(float[] stateVector) {
        if (stateVector == null) {
            throw new IllegalArgumentException("stateVector is null");
        }
        if (stateVector.length != STATE_SIZE) {
            throw new IllegalArgumentException(
                    "Expected state size " + STATE_SIZE + ", got " + stateVector.length
            );
        }

        // 1) Encode state through shared backbone
        INDArray stateArray = Nd4j.create(stateVector).reshape(1, STATE_SIZE);
        cachedFeatures = sharedEncoder.output(stateArray, false);

        // 2) Run all heads (each head should end in SOFTMAX if it's a categorical distribution)
        cachedMainActionProbs = mainActionHead.output(cachedFeatures, false).toFloatVector();
        cachedTakeMaterialProbs = takeMaterialHead.output(cachedFeatures, false).toFloatVector();
        cachedRefillDeckProbs = refillDeckHead.output(cachedFeatures, false).toFloatVector();
        cachedTicketSelectionProbs = ticketSelectionHead.output(cachedFeatures, false).toFloatVector();
        cachedFleetCompositionProbs = fleetCompositionHead.output(cachedFeatures, false).toFloatVector();
        cachedColorPreferenceProbs = colorPreferenceHead.output(cachedFeatures, false).toFloatVector();
        cachedRessourceValueProbs = resourceValueHead.output(cachedFeatures, false).toFloatVector();
        cachedDoubleBiasProbs = doubleBiasHead.output(cachedFeatures, false).toFloatVector();
        cachedTrackSelectionProbs = trackSelectionHead.output(cachedFeatures, false).toFloatVector();


    }

    /**
     * Optional: clear cached outputs (not required, but useful for debugging).
     */
    public void clearCache() {
        cachedMainActionProbs = null;
        cachedTakeMaterialProbs = null;
        cachedRefillDeckProbs = null;
        cachedTicketSelectionProbs = null;
        cachedFleetCompositionProbs = null;
        cachedColorPreferenceProbs = null;
        cachedRessourceValueProbs = null;
        cachedDoubleBiasProbs = null;
        cachedTrackSelectionProbs = null;

    }

// =======================
// GETTERS FOR HEAD OUTPUTS
// =======================

    /**
     * Main action distribution (routes, draw tickets, draw cards, etc.).
     * If legalMask != null, illegal actions are zeroed and the distribution renormalized.
     * <p>
     * NOTE: evaluate(state) must be called first.
     */
    public float[] getCachedMainActionProbs(boolean[] legalMask) {
        return (legalMask == null) ? cachedMainActionProbs.clone() : applyMask(cachedMainActionProbs, legalMask);
    }

    public float[] getCachedTakeMaterialProbs(boolean[] legalMask) {
        return (legalMask == null) ? cachedTrackSelectionProbs.clone() : applyMask(cachedTrackSelectionProbs, legalMask);
    }

    public float[] getCachedRefillDeckProbs(boolean[] legalMask) {
        return (legalMask == null) ? cachedRefillDeckProbs.clone() : applyMask(cachedRefillDeckProbs, legalMask);
    }

    public float[] getCachedTicketSelectionProbs(boolean[] legalMask) {
        return (legalMask == null) ? cachedTicketSelectionProbs.clone() : applyMask(cachedTicketSelectionProbs, legalMask);
    }

    public float[] getCachedFleetCompositionProbs(boolean[] legalMask) {
        return (legalMask == null) ? cachedFleetCompositionProbs.clone() : applyMask(cachedFleetCompositionProbs, legalMask);
    }

    public float[] getCachedColorPreferenceProbs(boolean[] legalMask) {
        return (legalMask == null) ? cachedColorPreferenceProbs.clone() : applyMask(cachedColorPreferenceProbs, legalMask);
    }

    public float[] getCachedRessourceValueProbs(boolean[] legalMask) {
        return (legalMask == null) ? cachedRessourceValueProbs.clone() : applyMask(cachedRessourceValueProbs, legalMask);
    }

    public float[] getCachedDoubleBiasProbs(boolean[] legalMask) {
        return (legalMask == null) ? cachedDoubleBiasProbs.clone() : applyMask(cachedDoubleBiasProbs, legalMask);
    }

    public float[] getCachedTrackSelectionProbs(boolean[] legalMask) {
        return (legalMask == null) ? cachedTrackSelectionProbs.clone() : applyMask(cachedTrackSelectionProbs, legalMask);
    }







// =======================
// INTERNAL HELPERS
// =======================

    private void ensureEvaluated(float[] cached, String headName) {
        if (cached == null) {
            throw new IllegalStateException(
                    "Head '" + headName + "' is not available. Did you forget to call evaluate(stateVector) first?"
            );
        }
    }


    /**
     * Example usage / training loop
     * <p>
     * TODO: You need to implement:
     * - GameState.toVector() - convert game state to float[300]
     * - GameState.getLegalMainActions() - return boolean[90] for legal actions
     * - GameState.getLegalSecondCardActions() - return boolean[8] for legal second card draws
     * - Game simulation loop
     */
   /* public static void main(String[] args) throws IOException {
        TicketToRideAgent agent = new TicketToRideAgent();

        // Try to load existing model
        File modelDir = new File("model_checkpoint");
        if (modelDir.exists()) {
            agent.loadModel("model_checkpoint");
            System.out.println("Loaded existing model");
        } else {
            System.out.println("Starting with fresh model");
        }

        List<Experience> batch = new ArrayList<>();

        // Training loop
        for (int game = 0; game < 10000; game++) {
            // TODO: Implement your game simulation here
            // Example structure:

            game ticketToRide = new game();
            List<Experience> gameExperiences = new ArrayList<>();

            while (!ticketToRide.isOver) {
                GameState state = ticketToRide.getCurrentState();
                float[] stateVector = state.toVector();
                boolean[] legalMask = state.getLegalMainActions();

                // Get main action
                float[] mainProbs = agent.getMainActionProbabilities(stateVector, legalMask);
                int mainAction = agent.sampleAction(mainProbs);

                // Handle card drawing (2 calls to agent)
                if (mainAction >= DRAW_TRAIN_DECK && mainAction <= DRAW_FACEUP_END) {
                    ticketToRide.drawCard(mainAction);

                    // Check if turn continues (joker from face-up ends turn)
                    if (!ticketToRide.drewJokerFromFaceUp()) {
                        GameState stateAfterFirst = ticketToRide.getCurrentState();
                        boolean[] secondCardMask = stateAfterFirst.getLegalSecondCardActions();

                        float[] secondProbs = agent.getSecondCardProbabilities(
                                stateAfterFirst.toVector(), false, secondCardMask);
                        int secondCard = agent.sampleAction(secondProbs);

                        ticketToRide.drawCard(secondCard);
                        gameExperiences.add(new Experience(stateVector, mainAction, -1, secondCard, 0));
                    } else {
                        gameExperiences.add(new Experience(stateVector, mainAction, -1, -1, 0));
                    }
                }
                // Handle fleet rebalancing
                else if (mainAction == REBALANCE_FLEET) {
                    float[] fleetProbs = agent.getFleetCompositionProbabilities(stateVector);
                    int locoCount = 10 + agent.sampleAction(fleetProbs);
                    ticketToRide.rebalanceFleet(locoCount);
                    gameExperiences.add(new Experience(stateVector, mainAction, locoCount - 10, -1, 0));
                }
                // Handle other actions
                else {
                    ticketToRide.executeAction(mainAction);
                    gameExperiences.add(new Experience(stateVector, mainAction, -1, -1, 0));
                }
            }

            // Assign rewards
            float reward = ticketToRide.didPlayerWin(0) ? 1.0f : -1.0f;
            for (Experience exp : gameExperiences) {
                exp.reward = reward;
            }
            batch.addAll(gameExperiences);


            // Train every 64 games
            if (game % 64 == 0 && !batch.isEmpty()) {
                agent.trainOnBatch(batch);
                batch.clear();
                System.out.println("Trained on batch at game " + game);
            }

            // Save checkpoint every 1000 games
            if (game % 1000 == 0 && game > 0) {
                agent.saveModel("model_checkpoint");
                System.out.println("Checkpoint saved at game " + game);
            }
        }

        // Save final model
        agent.saveModel("model_final");
        System.out.println("Training complete!");
    }*/
}