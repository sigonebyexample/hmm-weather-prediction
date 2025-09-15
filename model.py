import numpy as np

class HiddenMarkovModel:
    def __init__(self, transitions, emissions, starts, state_names):
        self.transitions = transitions  # Transition matrix
        self.emissions = emissions      # Emission probabilities
        self.starts = starts            # Initial state probabilities
        self.state_names = state_names  # State names
        self.n_states = len(state_names)
    
    def predict(self, observations):
        """
        Predict the most likely sequence of hidden states using Viterbi algorithm
        """
        # Convert observations to indices
        obs_map = {"umbrella": 0, "no umbrella": 1}
        obs_indices = [obs_map[obs] for obs in observations]
        
        n_obs = len(obs_indices)
        
        # Initialize Viterbi matrices
        viterbi = np.zeros((self.n_states, n_obs))
        backpointer = np.zeros((self.n_states, n_obs), dtype=int)
        
        # Initialization step
        for state in range(self.n_states):
            viterbi[state, 0] = self.starts[state] * self.emissions[state][obs_indices[0]]
            backpointer[state, 0] = 0
        
        # Recursion step
        for t in range(1, n_obs):
            for state in range(self.n_states):
                max_prob = -1
                max_prev_state = 0
                
                for prev_state in range(self.n_states):
                    prob = viterbi[prev_state, t-1] * self.transitions[prev_state, state] * self.emissions[state][obs_indices[t]]
                    if prob > max_prob:
                        max_prob = prob
                        max_prev_state = prev_state
                
                viterbi[state, t] = max_prob
                backpointer[state, t] = max_prev_state
        
        # Termination step
        best_path_prob = -1
        best_last_state = 0
        for state in range(self.n_states):
            if viterbi[state, n_obs-1] > best_path_prob:
                best_path_prob = viterbi[state, n_obs-1]
                best_last_state = state
        
        # Backtrack to find the best path
        best_path = [best_last_state]
        for t in range(n_obs-1, 0, -1):
            best_path.insert(0, backpointer[best_path[0], t])
        
        return best_path

# Create and export the weather prediction model
def create_weather_model():
    # Transition matrix: [sun->sun, sun->rain], [rain->sun, rain->rain]
    transitions = np.array([
        [0.8, 0.2],  # Tomorrow's predictions if today = sun
        [0.3, 0.7]   # Tomorrow's predictions if today = rain
    ])

    # Emission probabilities: [P(umbrella|state), P(no umbrella|state)]
    emissions = [
        [0.2, 0.8],  # sun: P(umbrella|sun)=0.2, P(no umbrella|sun)=0.8
        [0.9, 0.1]   # rain: P(umbrella|rain)=0.9, P(no umbrella|rain)=0.1
    ]

    # Initial state probabilities
    starts = np.array([0.5, 0.5])  # P(sun)=0.5, P(rain)=0.5

    # State names
    state_names = ["sun", "rain"]

    return HiddenMarkovModel(transitions, emissions, starts, state_names)

# For testing the model directly
if __name__ == "__main__":
    model = create_weather_model()
    print("Weather HMM model created successfully!")
