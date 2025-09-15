from model import create_weather_model

def predict_weather_sequence(observations):
    """
    Predict weather sequence based on umbrella observations
    """
    model = create_weather_model()
    predictions = model.predict(observations)
    
    result = []
    for prediction in predictions:
        result.append(model.state_names[prediction])
    
    return result

# Main execution
if __name__ == "__main__":
    # Observed data
    observations = [
        "umbrella",
        "umbrella",
        "no umbrella",
        "umbrella",
        "umbrella",
        "umbrella",
        "umbrella",
        "no umbrella",
        "no umbrella"
    ]

    print("Observations:", observations)
    
    # Predict underlying states
    weather_predictions = predict_weather_sequence(observations)
    
    print("\nPredicted weather sequence:")
    for i, (obs, weather) in enumerate(zip(observations, weather_predictions)):
        print(f"Day {i+1}: {obs} -> {weather}")
    
    print(f"\nFull sequence: {weather_predictions}")
