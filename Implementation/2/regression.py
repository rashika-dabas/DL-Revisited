import torch
from torch import nn


def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    Use nn.Linear
    """
    model = nn.Linear(input_size, output_size)

    return model

def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def fit_regression_model(X, y):
    """
    Train the model for the given number of epochs.
    Use the train_iteration function.
    While working, print the loss every 1000 epochs.
    Use the previous_loss variable to stop the training when the loss is not changing much.
    """

    # Hyperparameters
    learning_rate = 0.001  # Adjusted learning rate
    num_epochs = 10000     # Adjusted number of epochs
    tolerance = 0.01       # Tolerance for early stopping

    input_features = X.shape[1]  # Extract number of input features
    output_features = y.shape[1] # Extract number of output features

    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    prev_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        
        # Early stopping condition
        if abs(prev_loss - loss.item()) < tolerance:
            break
        
        prev_loss = loss.item()

        # Print the loss every 1000 epochs
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    return model, loss
