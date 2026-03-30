public class gamestate {
    game game;
    float[] inVec;

    public gamestate(game game) {
        this.game = game;
        inVec = new float[1000];
    }

    public void update() {

    }

    private void updateTrackOwnership() {
        int current_player_move = game.getCurrent_player_move();
        boolean[][] ownership = game.getTracks_ownership();
        for (int i = 0; i != 4; i++) {
            int actualPlayer = (current_player_move+i) %4;
            for (int j = 0; j != 129; j++) {
                inVec[i * 129 + j] = ownership[actualPlayer][j] ? 1.0f : 0.0f;
            }
        }
    }
    private void updateHarbors() {
        int pointer = 713;
        int current_player_move = game.getCurrent_player_move();
        boolean[][] harbors = game.getHarbors();
        for (int i = 0; i != 4; i++) {
            int actualPlayer = (current_player_move+i) %4;
            for (int j = 0; j != 37; j++) {
                inVec[pointer] = harbors[actualPlayer][j] ? 1.0f : 0.0f;
                pointer++;
            }
        }
    }
    private void updateUnKnownCards(){
        int current_player_move = game.getCurrent_player_move();
        int[][] knownCards = game.getKnownCards();
        int pointer =861;
        for (int i = 0;i!=4;i++){
            int actualPlayer = (current_player_move+i)%4;
            for (int j = 0;j!=6;j++){
                inVec[pointer]=( (float) knownCards[actualPlayer][j]) /6;
                pointer++;
            }
            for (int j = 6;j!=12;j++){
                inVec[pointer] = ((float)knownCards[actualPlayer][j])/4;
                pointer++;
            }
            for (int j =12;j!=18;j++){
                inVec[pointer]= ((float)knownCards[actualPlayer][j]/7);
                pointer++;
            }
            for (int j =18;j!=24;j++){
                inVec[pointer]= ((float)knownCards[actualPlayer][j]/4);
                pointer++;
            }
            inVec[pointer]=((float) knownCards[actualPlayer][24]/14);
            pointer++;
        }
        int[][]unknown = game.getUnknownCards();
        for (int i = 0;i!=4;i++){
            int actualPlayer = (current_player_move+i)%4;
                inVec[pointer++]=((float)unknown[actualPlayer][0])/60;
                inVec[pointer++]=((float)unknown[actualPlayer][1])/80;
        }
    }
    private void updateDiscard() {
        int pointer = 973;
        int[] discard = game.getDiscardStack();

        // Colors 0-5
        for (int j = 0; j < 6; j++) {
            inVec[pointer++] = discard[j] / 6.0f;
        }
        // Colors 6-11
        for (int j = 6; j < 12; j++) {
            inVec[pointer++] = discard[j] / 4.0f;
        }
        // Colors 12-17
        for (int j = 12; j < 18; j++) {
            inVec[pointer++] = discard[j] / 7.0f;
        }
        // Colors 18-23
        for (int j = 18; j < 24; j++) {
            inVec[pointer++] = discard[j] / 4.0f;
        }
        // Index 24 (Jokers)
        inVec[pointer] = discard[24] / 14.0f;
    }


    private void UpdateMisc() {
        inVec[516] = game.firstRound ? 1.0f : 0.0f; // firstround
        switch (game.getRemaining_rounds()){        //lastround
            case 0: inVec[517] = 0;break;
            case 1: inVec[517]= 0.3f; break;
            case 2: inVec[517]= 0.6f; break;
            default: inVec[517] = 1.0f; break;
        }
        for (int i = 0;i!=65;i++){
            inVec[518+i]= game.getDestinationFullfillment()[game.getCurrent_player_move()][i];
        }//destination fullfillment
    }
    private void insertDestBonusMalus(){
        //TODO when tickets created
    }
    private void stacks(){
        int pointer = 969;
        inVec[pointer++]=(float) game.getDiscardBoats()/60f;
        inVec[pointer++]=(float) game.getDiscardLokos()/80f;
        inVec[pointer++]=(float) game.getRandomBoats()/60f;
        inVec[pointer]=(float) game.getRandomLokos()/80f;
    }


    public float[] toVector() {
        return new float[700];//TODO ADD REAL NUMBER IN INITIALISTION OF GAMESTATE
    }
}


//0-128 track ownership self bool
//129-257 op 1 track ownership
//258-386 op2 track ownership
//387-515 op 3 track ownership
//516 firstround bool
//517 last/second last round scaled, 1 if running, 0.6 if 2 remaining, 0.3 if 1, 0 if 0
//518-582 destination completion/ownership 0 if not picked, 0.3 if picked. 0.6 for tour destination partly, 1.0 for complete
//583-647 destination reward final scaled from max ticket value
//648-712 destination malus final scaled form max ticket value
//713-749 harbors self bool
//750-786 harbor op1
//787-823 harbor op2
//824-860 harbors op3
//861-960 knownCards
//961-968 unkownCards
//969-970 discard stacks
//971-972 random stacks
//973-997 known used discarded cards


