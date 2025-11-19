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

    // Network architecture
    private MultiLayerNetwork sharedEncoder;
    private MultiLayerNetwork mainActionHead;
    private MultiLayerNetwork fleetCompositionHead;
    private MultiLayerNetwork secondCardHead;// For choosing the 2nd card when drawing
    private MultiLayerNetwork boatUsageHead;

    // ===== ADJUST THESE IF YOUR STATE ENCODING CHANGES =====
    private static final int STATE_SIZE = 645; // TODO: Change based on your state encoding size

    // ===== ADJUST THESE IF YOUR GAME BOARD CHANGES =====
    private static final int NUM_ROUTES = 129; // TODO: Change to match number of routes on your map
    private static final int NUM_FACE_UP_CARDS = 6; // 6 face-up cards in Rails & Sails

    private static final int NUM_BOAT_MODES = 5;

    // Calculated action space sizes (don't change these)
    private static final int NUM_DRAW_ACTIONS = 2 + NUM_FACE_UP_CARDS; // 2 decks + 6 face-up = 8
    private static final int NUM_MAIN_ACTIONS = NUM_DRAW_ACTIONS + NUM_ROUTES + 2; // Draw(8) + Routes(80) + DrawDest(1) + Rebalance(1) = 90
    private static final int NUM_FLEET_OPTIONS = 14;

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
    public static final int REBALANCE_FLEET = 177;   // Rebalance locomotive/boat split

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
                        .nIn(STATE_SIZE)  // TODO: Must match your state vector size
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

        secondCardHead = new MultiLayerNetwork(secondCardConf);
        secondCardHead.init();

        // ===== BOAT USAGE HEAD - Decides how many boats to spend when claiming a route =====
        MultiLayerConfiguration boatUsageConf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(128)      // same as shared encoder output size
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(64)
                        .nOut(NUM_BOAT_MODES)   // 3 modes: min / medium / max
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        boatUsageHead = new MultiLayerNetwork(boatUsageConf);
        boatUsageHead.init();
    }

    /**
     * Get main action probabilities from state
     *
     * @param state     Float array representing game state (must be STATE_SIZE length)
     *                               TODO: You need to implement GameState.toVector() to create this
     * @param legalMask Boolean array marking which actions are legal (same length as NUM_MAIN_ACTIONS)
     *                                   TODO: You need to implement GameState.getLegalMainActions()
     * @return Probability distribution over all main actions
     */
    public float[] getMainActionProbabilities(float[] state, boolean[] legalMask) {
        INDArray stateArray = Nd4j.create(state).reshape(1, STATE_SIZE);

        // Encode state through shared layers
        INDArray features = sharedEncoder.output(stateArray);

        // Get action probabilities
        INDArray output = mainActionHead.output(features);
        float[] probs = output.toFloatVector();

        // Apply legal action mask (zeros out illegal actions and renormalizes)
        if (legalMask != null) {
            probs = applyMask(probs, legalMask);
        }

        return probs;
    }

    /**
     * Get second card draw probabilities
     *
     * @param state               Current game state after first card was drawn
     * @param drewJokerFromFaceUp True if first card was a joker from face-up cards
     *                            (means turn ends immediately, this shouldn't be called)
     * @param legalMask           Boolean array for legal second draws (8 elements: 2 decks + 6 face-up)
     *                                             TODO: Implement logic to mark jokers in face-up as illegal if first was joker
     * @return Probability distribution over second card choices
     */
    public float[] getSecondCardProbabilities(float[] state, boolean drewJokerFromFaceUp, boolean[] legalMask) {
        // If drew joker from face-up on first draw, turn ends (no second draw)
        if (drewJokerFromFaceUp) {
            return new float[NUM_DRAW_ACTIONS]; // All zeros
        }

        INDArray stateArray = Nd4j.create(state).reshape(1, STATE_SIZE);
        INDArray features = sharedEncoder.output(stateArray);
        INDArray output = secondCardHead.output(features);
        float[] probs = output.toFloatVector();

        // Apply legal mask (e.g., can't draw second joker from face-up)
        if (legalMask != null) {
            probs = applyMask(probs, legalMask);
        }

        return probs;
    }

    /**
     * Get fleet composition probabilities (10-30 locomotives)
     */
    public float[] getFleetCompositionProbabilities(float[] state) {
        INDArray stateArray = Nd4j.create(state).reshape(1, STATE_SIZE);
        INDArray features = sharedEncoder.output(stateArray);
        INDArray output = fleetCompositionHead.output(features);
        return output.toFloatVector();
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
     * Get boat usage probabilities for route claiming.
     * Only called when mainAction is a CLAIM_ROUTE_* action.
     *
     * @param state Float array representing game state (STATE_SIZE)
     * @return Probability distribution over boat usage modes:
     *         0 = minimal boats, 1 = medium, 2 = max boats
     */
    public float[] getBoatUsageProbabilities(float[] state) {
        INDArray stateArray = Nd4j.create(state).reshape(1, STATE_SIZE);
        INDArray features = sharedEncoder.output(stateArray);
        INDArray output = boatUsageHead.output(features);
        return output.toFloatVector();
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
        secondCardHead.fit(dataSet);
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
        boatUsageHead.fit(dataSet);
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
        secondCardHead.save(new File(dir, "second_card_head.zip"));

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
        secondCardHead = MultiLayerNetwork.load(new File(dir, "second_card_head.zip"), true);

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
        public float reward;           // Reward for this experience

        public Experience(float[] state,
                          int mainAction,
                          int fleetAction,
                          int secondCardAction,
                          int boatUsageAction,
                          float reward) {
            this.state = state;
            this.mainAction = mainAction;
            this.fleetAction = fleetAction;
            this.secondCardAction = secondCardAction;
            this.boatUsageAction = boatUsageAction;
            this.reward = reward;
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
    public static void main(String[] args) throws IOException {
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
    }
}