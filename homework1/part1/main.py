import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class MyLinearRegression:
    def __init__(
        self,
        weights_init="random",
        add_bias=True,
        learning_rate=1e-5,
        num_iterations=1_000,
        verbose=False,
        max_error=1e-5,
    ):
        """Linear regression model using gradient descent"""
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.weights_init = weights_init
        self.add_bias = add_bias
        self.verbose = verbose
        self.max_error = max_error

    def initialize_weights(self, n_features):
        """Initialize weights"""
        if self.weights_init == "random":
            ################
            weights = np.random.randn(n_features, 1)
            ################
        elif self.weights_init == "zeros":
            ################
            weights = np.zeros((n_features, 1))
            ################
        else:
            raise NotImplementedError
        return weights

    def cost(self, target, pred):
        """Calculate mean squared error"""
        ################
        loss = np.mean((target - pred) ** 2)
        ################
        return loss

    def fit(self, x, y):
        if self.add_bias:
            ################
            x = np.c_[np.ones((x.shape[0], 1)), x]
            ################

        self.weights = self.initialize_weights(x.shape[1])

        for i in range(self.num_iterations):
            ################
            y_pred = x @ self.weights
            current_loss = self.cost(y, y_pred)
            gradient = -(2 / x.shape[0]) * x.T @ (y - y_pred)
            self.weights -= self.learning_rate * gradient
            new_loss = self.cost(y, x @ self.weights)

            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}: Loss = {new_loss}")

            if abs(new_loss - current_loss) < self.max_error:
                break
            ################

    def predict(self, x):
        """prediction function"""
        ################
        if self.add_bias:
            x = np.c_[np.ones((x.shape[0], 1)), x]
        y_hat = x @ self.weights
        ################
        return y_hat


def normal_equation(X, y):
    """Calculate weights using the normal equation"""
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y


if __name__ == "__main__":
    # Generating data samples
    x = np.linspace(-5.0, 5.0, 100)[:, np.newaxis]
    y = 29 * x + 40 * np.random.rand(100, 1)

    # Normalizing input data
    x /= np.max(x)

    plt.title("Data samples")
    plt.scatter(x, y)
    plt.savefig("data_samples.png")

    # Sklearn linear regression model
    sklearn_model = LinearRegression()
    sklearn_model.fit(x, y)
    y_hat_sklearn = sklearn_model.predict(x)

    plt.title("Data samples with sklearn model")
    plt.scatter(x, y)
    plt.plot(x, y_hat_sklearn, color="r")
    plt.savefig("sklearn_model.png")
    print("Sklearn MSE: ", mean_squared_error(y, y_hat_sklearn))

    # Your linear regression model
    my_model = MyLinearRegression(verbose=False, num_iterations=1_000_000)
    my_model.fit(x, y)
    y_hat = my_model.predict(x)

    plt.title("Data samples with my model")
    plt.scatter(x, y)
    plt.plot(x, y_hat, color="r")
    plt.savefig("my_model.png")
    print("My MSE: ", mean_squared_error(y, y_hat))

    # Normal equation
    weights = normal_equation(x, y)
    y_hat_normal = np.c_[np.ones((x.shape[0], 1)), x] @ weights

    plt.title("Data samples with normal equation")
    plt.scatter(x, y)
    plt.plot(x, y_hat_normal, color="r")
    plt.savefig("normal_equation.png")
    print("Normal equation MSE: ", mean_squared_error(y, y_hat_normal))
