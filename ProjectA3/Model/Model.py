import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder

class MyLogisticRegression:

    def __init__(self, cv: int = 5, lr: float = 0.1,
                 max_iter: int = 5000, weight_init: str = 'uniform',
                 method: str = 'mini_batch', batch_size: int = 64, l2: float = None, ):
        self.lr = lr
        self.cv = cv
        self.max_iter = max_iter
        self.weight_init = weight_init
        self.method = method
        self.batch_size = batch_size
        self.l2 = 0 if l2 is None else l2

        valid_weight_init = ['uniform', 'normal', 'xavier', 'ones']

        if weight_init not in valid_weight_init:
            raise ValueError(f'weight_init must be one of {valid_weight_init}')

        valid_method = ['mini_batch', 'batch', 'stochastic']

        if method not in valid_method:
            raise ValueError(f'method must be one of {valid_method}')

    def fit(self, X, y):
        self.split = KFold(n_splits=self.cv)
        y_class = len(np.unique(y))

        if hasattr(X, "toarray"):
            X = X.toarray()  # Convert sparse matrix to dense

        # Ensure y is a NumPy array
        if not isinstance(y, np.ndarray):
            y = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)

        # One-Hot Encode y
        self.oh = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        y_encoded = self.oh.fit_transform(y.reshape(-1, 1))

        self.losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []

        for fold, (train_index, test_index) in enumerate(self.split.split(X)):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y_encoded[train_index], y_encoded[test_index]

            X_train = self._add_intercept(X_train)
            X_val = self._add_intercept(X_val)

            self.W = self.weight_initializer(X_train, y_class)
            self.velocity = np.zeros_like(self.W)

            fold_train_losses = []
            fold_train_accuracies = []

            if self.method == 'mini_batch':
                for i in range(self.max_iter):
                    ix = np.random.randint(0, X_train.shape[0])
                    X_train_batch = X_train[ix:ix + self.batch_size]
                    y_train_batch = y_train[ix:ix + self.batch_size]
                    loss = self.train(X_train_batch, y_train_batch)
                    fold_train_losses.append(loss)

                    _, train_pred = self.predict(X_train_batch)
                    train_accuracy = np.mean(np.argmax(y_train_batch, axis=1) == train_pred)
                    fold_train_accuracies.append(train_accuracy)

                    if i % 500 == 0:
                        print(f"Iteration {i} - Loss: {loss:.4f}, Accuracy: {train_accuracy:.4f}")

            elif self.method == 'batch':
                for i in range(self.max_iter):
                    loss = self.train(X_train, y_train)
                    fold_train_losses.append(loss)

                    _, train_pred = self.predict(X_train)
                    train_accuracy = np.mean(np.argmax(y_train, axis=1) == train_pred)
                    fold_train_accuracies.append(train_accuracy)

                    if i % 500 == 0:
                        print(f"Iteration {i} - Loss: {loss:.4f}, Accuracy: {train_accuracy:.4f}")

            elif self.method == 'stochastic':
                for i in range(self.max_iter):
                    idx = np.random.randint(X_train.shape[0])  # Select a random index
                    X_sto = X_train[idx, :].reshape(1, -1)  # Get the single sample
                    y_sto = y_train[idx].reshape(1, -1)  # Get the corresponding label

                    loss = self.train(X_sto, y_sto)  # Train the model on this single example

                    if not np.isnan(loss):  # Ensure the loss is valid
                        fold_train_losses.append(loss)

                    _, train_pred = self.predict(X_sto)  # Get predicted class
                    train_accuracy = np.mean(np.argmax(y_sto, axis=1) == train_pred)
                    fold_train_accuracies.append(train_accuracy)

                    if i % 500 == 0:
                        print(f"Iteration {i} - Loss: {loss:.4f}, Accuracy: {train_accuracy:.4f}")

            # Store the average training loss & accuracy for this fold
            avg_train_loss = np.mean(fold_train_losses)
            avg_train_accuracy = np.mean(fold_train_accuracies)
            self.losses.append(avg_train_loss)
            self.train_accuracies.append(avg_train_accuracy)

            val_pred = self.predict(X_val)[1]  # Get predicted class labels
            y_val_labels = self.oh.inverse_transform(y_val)

            # Compute validation loss
            val_loss = self.cross_entropy(self.softmax_(X_val @ self.W), y_val)
            self.valid_losses.append(val_loss)

            # Compute validation accuracy
            val_accuracy = np.mean(y_val_labels.flatten() == val_pred)
            self.valid_accuracies.append(val_accuracy)

            print(f"Fold: {fold}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    def train(self, X, y):
        y_hat, _ = self.predict(X)

        error = y_hat - y
        m = max(X.shape[0], 1)  # Ensure division is valid

        loss = self.cross_entropy(y_hat, y) if m > 1 else float(np.mean(y_hat))

        grad = X.T @ error + 2 * self.l2 * self.W
        self.velocity = 0.8 * self.velocity - self.lr * grad
        self.W += self.velocity
        # self.W -= self.lr * grad

        return loss if not np.isnan(loss) else 0.0

    def predict(self, X, is_test=False):
        if is_test:
            X = self._add_intercept(X)

        y_hat = X @ self.W
        y_hat = self.softmax_(y_hat)
        y_real = np.argmax(y_hat, axis=1)
        return y_hat, y_real

    def softmax_(self, X):
        X_max = np.max(X, axis=1, keepdims=True)  # Find max per row
        exp_shifted = np.exp(X - X_max)  # Shift values for numerical stability
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)  # Normalize

    def weight_initializer(self, X, num_classes):
        if self.weight_init == 'uniform':
            return np.random.uniform(low=-self.lr, high=self.lr, size=(X.shape[1], num_classes))
        elif self.weight_init == 'normal':
            return np.random.randn(X.shape[1], num_classes)
        elif self.weight_init == 'xavier':
            limit = np.sqrt(6 / (X.shape[1] + num_classes))
            return np.random.uniform(low=-limit, high=limit, size=(X.shape[1], num_classes))
        else:
            return np.ones((X.shape[1], num_classes))

    def cross_entropy(self, y, y_hat):
        if y_hat.size == 0 or y.size == 0:
            return 0.0  # Return zero loss to avoid NaN issues

        m = max(y.shape[0], 1)  # Prevent division by zero
        loss = - np.sum(y * np.log(y_hat + 1e-9)) / m  ## Prevent log(0)
        return loss + self.l2 * np.sum(self.W ** 2)

    def _add_intercept(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def classification_report(self, pred, y):
        # Ensure both pred and y are NumPy arrays
        pred = np.array(pred).flatten()
        y = np.array(y)

        if len(pred) == 0 or len(y) == 0:  # Prevent empty array issues
            return 0.0  # Return zero accuracy to avoid NaN

        # Extract unique labels from both predictions and ground truth
        labels = np.unique(np.concatenate((y, pred)))
        num_classes = len(labels)  # Update num_classes based on unique labels

        # Initialize confusion matrix
        cm = np.zeros((num_classes, num_classes), dtype=int)

        for true, pred_label in zip(y, pred):
            # Get index positions of true and predicted labels in `labels`
            true_idx = np.where(labels == true)[0]
            pred_idx = np.where(labels == pred_label)[0]

            if true_idx.size > 0 and pred_idx.size > 0:  # Check if valid indices exist
                cm[true_idx[0], pred_idx[0]] += 1

        # Compute metrics
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp

        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
        f1_score = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision, dtype=float),
                             where=(precision + recall) > 0)

        accuracy_score = np.sum(np.diag(cm)) / np.sum(cm) if np.sum(cm) > 0 else 0.0

        # Weighted metrics
        class_counts = np.array([(y == label).sum() for label in labels]) / len(y)
        weighted_precision = np.sum(class_counts * precision)
        weighted_recall = np.sum(class_counts * recall)
        weighted_f1 = np.sum(class_counts * f1_score)

        # Print report
        print("\nClassification Report:\n")
        print("{:<10} {:<10} {:<10} {:<10}".format("Class", "Precision", "Recall", "F1-score"))
        print("-" * 40)
        for i, label in enumerate(labels):
            print(f"{label:<10} {precision[i]:<10.2f} {recall[i]:<10.2f} {f1_score[i]:<10.2f}")

        return round(accuracy_score, 2), round(weighted_precision, 2), round(weighted_recall, 2), round(weighted_f1, 2)

    def _coeff_and_biases(self, feature_names):
        if not hasattr(self, "W"):
            raise ValueError("Model is not trained yet. Fit the model before retrieving coefficients.")

        coef = self.W[1:, :]  # Exclude bias term
        bias = self.W[0, :]
        print(coef.shape, len(feature_names))
        # Create a DataFrame for easy interpretation
        coef_df = pd.DataFrame(coef, index=feature_names,
                               columns=[f"Class_{i}" for i in range(coef.shape[1])] if coef.ndim > 1 else [
                                   "Coefficient"])

        print("\nTop Important Features:")
        print(coef_df.abs().sum(axis=1).sort_values(ascending=False).head(10))

        return coef_df, bias