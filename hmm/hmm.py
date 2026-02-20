import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        T = len(input_observation_states) # variable to store length of input sequence
        S = len(self.hidden_states) # number of hidden states
        alpha = np.zeros((T, S))

        # initialize row 0 of alpha
        obs_idx = self.observation_states_dict[input_observation_states[0]]
        for s in range(S):
            alpha[0][s] = self.prior_p[s] * self.emission_p[s][obs_idx]

        # Step 2. Calculate probabilities
        for t in range(1,T):
            obs_idx = self.observation_states_dict[input_observation_states[t]] # find proper idx
            for s in range(S):
                alpha[t][s] = sum(alpha[t-1][r]*self.transition_p[r][s]* self.emission_p[s][obs_idx]
                                  for r in range(S))

        # Step 3. Return final probability
        return np.sum(alpha[T-1]) # just sum over all hidden states at final timestep!


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        # store T = len observed and S = len hidden states for efficiency in code
        T = len(decode_observation_states)
        S = len(self.hidden_states)

        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((T, S))
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))         
        prev = np.empty((T, S))

        # Step 2. Calculate Probabilities
        obs_idx = self.observation_states_dict[decode_observation_states[0]]
        for s in range(S):
            viterbi_table[0][s] = self.prior_p[s] * self.emission_p[s][obs_idx]
        
        for t in range(1, T):
            obs_idx = self.observation_states_dict[decode_observation_states[t]]
            for s in range(S):
                for r in range(S):
                    new_prob = viterbi_table[t-1][r] * self.transition_p[r][s] * self.emission_p[s][obs_idx]
                    if new_prob > viterbi_table[t][s]:
                        viterbi_table[t][s] = new_prob
                        prev[t][s] = r
        
        # Step 3. Traceback 
        best_path[T-1] = np.argmax(viterbi_table[T-1])
        for t in range(T-2, -1, -1): # iterate backwards
            best_path[t] = prev[t+1][best_path[t+1]]
    
        # Step 4. Return best hidden state sequence
        return [self.hidden_states_dict[s] for s in best_path] # return names!
        