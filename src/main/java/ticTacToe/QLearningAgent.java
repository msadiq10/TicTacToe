package ticTacToe;

import java.util.List;
import java.util.Random;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is implemented in the {@link QTable} class.
 * 
 *  The methods to implement are: 
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object, in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to. 
 * @author ae187
 */

public class QLearningAgent extends Agent {
	
	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha=0.1;
	
	/**
	 * The number of episodes to train for
	 */
	int numEpisodes=50000;
	
	/**
	 * The discount factor (gamma)
	 */
	double discount=0.9;
	
	
	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon=0.1;
	
	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move) pair.
	 * 
	 */
	
	QTable qTable=new QTable();
	
	
	/**
	 * This is the Reinforcement Learning environment that this agent will interact with when it is training.
	 * By default, the opponent is the random agent which should make your q learning agent learn the same policy 
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env=new TTTEnvironment();
	
	
	/**
	 * Construct a Q-Learning agent that learns from interactions with {@code opponent}.
	 * @param opponent the opponent agent that this Q-Learning agent will interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from your lectures.
	 * @param numEpisodes The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount)
	{
		env=new TTTEnvironment(opponent);
		this.alpha=learningRate;
		this.numEpisodes=numEpisodes;
		this.discount=discount;
		initQTable();
		train();
	}
	
	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 *  
	 */
	
	protected void initQTable()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
		{
			List<Move> moves=g.getPossibleMoves();
			for(Move m: moves)
			{
				this.qTable.addQValue(g, m, 0.0);
				//System.out.println("initing q value. Game:"+g);
				//System.out.println("Move:"+m);
			}
			
		}
		
	}
	
	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent()
	{
		this(new RandomAgent(), 0.1, 50000, 0.9);
		
	}
	
	
	/**
	 *  Implement this method. It should play {@code this.numEpisodes} episodes of Tic-Tac-Toe with the TTTEnvironment, updating q-values according 
	 *  to the Q-Learning algorithm as required. The agent should play according to an epsilon-greedy policy where with the probability {@code epsilon} the
	 *  agent explores, and with probability {@code 1-epsilon}, it exploits. 
	 *  
	 *  At the end of this method you should always call the {@code extractPolicy()} method to extract the policy from the learned q-values. This is currently
	 *  done for you on the last line of the method.
	 */
	
	public void train()
	{
		/* 
		 * YOUR CODE HERE
		 */
		for(int i=0; i<numEpisodes; i++) {
			while(!this.env.isTerminal()) {
				// for a game state g in environment env
				Game g = this.env.getCurrentGameState();
				// if g is terminal state, skip it.
				if(g.isTerminal()){
					continue;
				}
				// pick a move using epsilon greedy policy,
				// check helper method pickEpsMove(Game g) for implementation
				Move m = pickEpsMove(g);
				Outcome outcome=null;
				try {
					// outcome after executing a move
					outcome = this.env.executeMove(m);
				} catch (IllegalMoveException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				// current q-value
				double qvalue = this.qTable.getQValue(outcome.s, outcome.move);
				double newqvalue;
				// updated Q(g, m) = (1 - alpha) * old Q(g, m) + alpha * (reward + discount * maxQvalue(g'))
				newqvalue = (1 - this.alpha)*qvalue + this.alpha*(outcome.localReward + this.discount*maxQvalue(outcome.sPrime));
				this.qTable.addQValue(outcome.s, outcome.move, newqvalue);
				
			}
			// reset after one iteration
		this.env.reset();
		}

		
		
		//--------------------------------------------------------
		//you shouldn't need to delete the following lines of code.
		this.policy=extractPolicy();
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			//System.exit(1);
		}
	}
	// helper method for epsilon greedy policy
	private Move pickEpsMove(Game g) {

		List<Move> moves = g.getPossibleMoves();	
		Move m = null;
		Random r = new Random();
		double qvalue = 0;
		double max = -Integer.MAX_VALUE;
		double random = r.nextDouble();
		
		if(random < epsilon) {	
			// if the random number is less than epsilon, it will explore
			if(moves.size()!=0) {
				Random random1 = new Random();
				int num = random1.nextInt(moves.size());													
				m = moves.get(num);	//Generate a random move m
			}

			
		}
		else {
			//Else, exploit
			// here, find move that gives maximum q-value
			for (Move m1 : moves){
				qvalue = qTable.getQValue(g, m1);
				if (qvalue>=max) {
					max = qvalue;
					m = m1;
				}
			}
			}
		
		return m; 
	}
	
	// helper method to get the Q-value of sPrime
	private Double maxQvalue(Game gPrime) {
		if(gPrime.isTerminal()){
			return 0.0;
		}
		double max = -Integer.MAX_VALUE;			
		double qvalue = 0;
		for (Move m : gPrime.getPossibleMoves()){
			
			qvalue = this.qTable.getQValue(gPrime, m);

			if (qvalue>max) {
				max = qvalue;
			}
		}
		return max;
	}
	
	/** Implement this method. It should use the q-values in the {@code qTable} to extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy()
	{
		// create new policy
		Policy p = new Policy();
		
		for(Game g : this.qTable.keySet()){
			if(g.isTerminal()){
				continue;
			}

			double max = -Integer.MAX_VALUE;
			Move maxMove = null;
			// iterate through all possible moves of g and
			// find a move that gives the max q-value
			for (Move m : g.getPossibleMoves()){
				
				double sum = qTable.getQValue(g, m);
				
				if (sum>max) {
					max = sum;
					maxMove = m;
				}
			}
			p.policy.put(g,maxMove); // set the move associated with the current state g to maxMove in policy. 
		}
		// return policy
		return p;
		
	}
	
	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play your agent against a human agent (yourself).
		QLearningAgent agent=new QLearningAgent();
		
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
	
	
	


	
}
