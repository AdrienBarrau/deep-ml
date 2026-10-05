import numpy as np

def behavior_clone(expert_states, expert_actions, test_states, lr, epochs):
    """
    Train a linear policy via behavior cloning on expert demonstrations,
    then predict actions for new states using pure NumPy.
    
    Args:
        expert_states: np.ndarray of shape (N, D) - expert state observations
        expert_actions: np.ndarray of shape (N,) - expert actions (scalar)
        test_states: np.ndarray of shape (M, D) - states to predict actions for
        lr: float - learning rate for gradient descent
        epochs: int - number of training iterations
    
    Returns:
        dict with:
            'predicted_actions': list of predicted actions for test states
            'final_loss': float, MSE on training data after training
    """
    N, D = expert_states.shape
    
    # Reshape expert_actions to (N, 1) to match matrix multiplication dimensions
    y_train = expert_actions.reshape(-1, 1)
    
    # Initialize weights and bias with zeros
    W = np.zeros((D, 1))
    b = np.zeros((1, 1))
    
    # Training Loop using Full-Batch Gradient Descent
    for epoch in range(epochs):
        # Forward pass: compute predictions
        predictions = np.dot(expert_states, W) + b
        
        # Compute error (residuals)
        error = predictions - y_train
        
        # Analytical gradients for Mean Squared Error
        dW = (2 / N) * np.dot(expert_states.T, error)
        db = (2 / N) * np.sum(error, axis=0, keepdims=True)
        
        # Update weights and bias
        W -= lr * dW
        b -= lr * db
        
    # Calculate final training loss
    final_predictions = np.dot(expert_states, W) + b
    final_loss = float(np.mean((final_predictions - y_train) ** 2))
    
    # Predict actions for test states
    test_predictions = np.dot(test_states, W) + b
    # Flatten array and convert to a list of scalars
    predicted_actions = test_predictions.flatten().tolist()
    
    return {
        'predicted_actions': predicted_actions,
        'final_loss': final_loss
    }
