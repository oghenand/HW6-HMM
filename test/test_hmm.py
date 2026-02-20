import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')
    #NOTE: Using mini hmm for forward testing, mini input for viterbi
    # first for forward
    # extract key vars
    hidden_states_f = mini_hmm['hidden_states']
    observation_states_f = mini_hmm['observation_states']
    prior_p_f = mini_hmm['prior_p']
    transition_p_f = mini_hmm['transition_p']
    emission_p_f = mini_hmm['emission_p']

    hmm = HiddenMarkovModel(observation_states_f, hidden_states_f,
                                    prior_p_f, transition_p_f, emission_p_f)
    ex_forward_inp = np.array(['sunny', 'rainy'])
    # first edge case: make sure observation states is at least 1!
    assert len(ex_forward_inp) >= 1, "At least one observation state must be passed in!"
    # run on observation states
    forward_probability = hmm.forward(ex_forward_inp)

    # assertions
    assert (forward_probability <= 1.0) and (forward_probability >=0.0), "Forward prob must be between 0 and 1!"
    # check from manual comparison
    assert np.isclose(forward_probability, 0.26625, atol=1e-6), "Forward probability is incorrect!"

    # now for viterbi
    # find hidden states for observation state sequence
    states_to_decode = mini_input['observation_state_sequence']
    true_best_path = mini_input['best_hidden_state_sequence']
    best_path = hmm.viterbi(states_to_decode)

    # assertions
    assert len(best_path) == len(states_to_decode), 'Best path must be of same len as input sequence'
    assert np.array_equal(best_path, true_best_path), "Best paths don't match!"
    assert all([val in hmm.hidden_states for val in best_path]), "All returned states must be hidden states!"

    # edge case for single-input viterbi
    single_result = hmm.viterbi(np.array(['sunny']))
    assert len(single_result) == 1, "single-len sequence should return result of length 1"


def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """
    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    hidden_states = full_hmm['hidden_states']
    observation_states = full_hmm['observation_states']
    prior_p = full_hmm['prior_p']
    transition_p = full_hmm['transition_p']
    emission_p = full_hmm['emission_p']

    # first test for forward
    hmm = HiddenMarkovModel(observation_states, hidden_states,
                                    prior_p, transition_p, emission_p)
    # generate random sequence
    ex_forward_inp = np.random.choice(observation_states, size=10)
    forward_probability = hmm.forward(ex_forward_inp)

    # make sure probability in valid range
    assert (forward_probability <= 1.0) and (forward_probability >=0.0), "Forward prob must be between 0 and 1!"

    # now test viterbi
    # find hidden states for observation state sequence
    states_to_decode = full_input['observation_state_sequence']
    true_best_path = full_input['best_hidden_state_sequence']
    best_path = hmm.viterbi(states_to_decode)

    # assertions
    assert len(best_path) == len(states_to_decode), 'Best path must be of same len as input sequence'
    assert np.array_equal(best_path, true_best_path), "Best paths don't match!"
    assert all([val in hmm.hidden_states for val in best_path]), "All returned states must be hidden states!"

    












