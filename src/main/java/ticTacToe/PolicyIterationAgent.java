package ticTacToe;


import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should runs/alternate (1) and (2) until convergence. 
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration: Convergence of the Values of the current policy, 
 * and Convergence of the current policy to the optimal policy.
 * The former happens when the values of the current policy no longer improve by much (i.e. the maximum improvement is less than 
 * some small delta). The latter happens when the policy improvement step no longer updates the policy, i.e. the current policy 
 * is already optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current policy (policy evaluation). 
	 */
	HashMap<Game, Double> policyValues=new HashMap<Game, Double>();
	
	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}. 
	 */
	HashMap<Game, Move> curPolicy=new HashMap<Game, Move>();
	
	double discount=0.9;
	
	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;
	
	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
		
		
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);
		
	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP paramters (rewards, transitions, etc) as specified in 
	 * {@link TTTMDP}
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		this.mdp=new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all states to 0 
	 * (V0 under some policy pi ({@link #curPolicy} from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.policyValues.put(g, 0.0);
		
	}
	
	/**
	 *  You should implement this method to initially generate a random policy, i.e. fill the {@link #curPolicy} for every state. Take care that the moves you choose
	 *  for each state ARE VALID. You can use the {@link Game#getPossibleMoves()} method to get a list of valid moves and choose 
	 *  randomly between them. 
	 */
	public void initRandomPolicy()
	{
		Random r = new Random();
		List<Move> moves; // list of all possible moves from game state g
		for (Game g : this.policyValues.keySet()){

			// if g is terminal, skip it
			if (g.isTerminal())
				continue;
			moves = g.getPossibleMoves(); // store the list of all possible moves from game state g in "moves"
			// put random moves in curPolicy for g
			this.curPolicy.put(g, moves.get(r.nextInt(moves.size())));
		}
	}
	
	
	
	/**
	 * Performs policy evaluation steps until the maximum change in values is less than {@code delta}, in other words
	 * until the values under the currrent policy converge. After running this method, 
	 * the {@link PolicyIterationAgent#policyValues} map should contain the values of each reachable state under the current policy. 
	 * You should use the {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 */
	protected void evaluatePolicy(double delta)
	{
		for (Game g : this.policyValues.keySet()){
			// if g is terminal skip it
			if (g.isTerminal()){
				this.policyValues.put(g, 0.0);
				continue;
			}
			double sum, prev;
			do {
				sum = 0;
				for (TransitionProb tp : this.mdp.generateTransitions(g, this.curPolicy.get(g))) {
					// q-value
					sum += tp.prob * (tp.outcome.localReward + (discount * this.policyValues.get(tp.outcome.sPrime)));
				}

				// Store the old value to update with the new value
				prev = this.policyValues.get(g);
				this.policyValues.put(g, sum);

				// until they converge
			} while (Math.abs(this.policyValues.get(g)-prev) > delta);

		}
	}
	
		
	
	
	/**This method should be run AFTER the {@link PolicyIterationAgent#evaluatePolicy} train method to improve the current policy according to 
	 * {@link PolicyIterationAgent#policyValues}. You will need to do a single step of expectimax from each game (state) key in {@link PolicyIterationAgent#curPolicy} 
	 * to look for a move/action that potentially improves the current policy. 
	 * 
	 * @return true if the policy improved. Returns false if there was no improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy()
	{
		// copy of original policy
		Policy copy = new Policy();
		for (Map.Entry<Game, Move> entry : this.curPolicy.entrySet())
		{
			copy.policy.put(entry.getKey(), entry.getValue());
		}

		// performing single step expectimax
		for(Game g : this.curPolicy.keySet()){
			double preValue = this.policyValues.get(g);
			for (Move m : g.getPossibleMoves()){
				double sum = 0;
				for(TransitionProb t : this.mdp.generateTransitions(g,m)){
					sum += t.prob*(t.outcome.localReward+(discount*this.policyValues.get(t.outcome.sPrime)));
				}

				// if sum greater than the previous value, then update.
				if (sum > preValue){
					preValue = sum;
					this.curPolicy.put(g,m);
				}
			}
		}
		// compare current policy with copy of last policy,
		// if no change, improvePolicy returns false, else true.
		if (this.curPolicy.equals(copy.policy))
			return false;
		else{
			return true;
		}
	}

	
	/**
	 * The (convergence) delta
	 */
	double delta=0.1;
	
	/**
	 * This method should perform policy evaluation and policy improvement steps until convergence (i.e. until the policy
	 * no longer changes), and so uses your 
	 * {@link PolicyIterationAgent#evaluatePolicy} and {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train()
	{
		// init random policy
		this.initRandomPolicy();
		// keep evaluating and improving policy 
		// until the policy stops improving
		do{
			this.evaluatePolicy(delta);
		}while(this.improvePolicy());
		

		// give new policy to super.policy
		Policy newPolicy = new Policy(curPolicy);
		super.policy = newPolicy;
	}
	
	public static void main(String[] args) throws IllegalMoveException
	{
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi=new PolicyIterationAgent();
		
		HumanAgent h=new HumanAgent();
		
		Game g=new Game(pi, h, h);
		
		g.playOut();
		
		
	}
	

}
