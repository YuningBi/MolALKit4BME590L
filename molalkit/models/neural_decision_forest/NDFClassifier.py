#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neural Decision Forest Classifier wrapper for MolALKit.

Wraps the PyTorch-based NeuralDecisionForest to provide a sklearn-like interface
compatible with MolALKit's active learning framework.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from molalkit.models.base import BaseModel


class NDFClassifier(BaseModel):
    """
    Neural Decision Forest Classifier for MolALKit.

    This wrapper provides sklearn-like fit/predict interface for the PyTorch-based
    NeuralDecisionForest model.

    Args:
        n_estimators: Number of trees in the forest (default: 10)
        tree_depth: Depth of each tree (default: 5, None for adaptive)
                   If None, depth is automatically set based on sample size:
                   - depth=6 for <500 samples (64 leaf nodes)
                   - depth=10 for ≥500 samples (1024 leaf nodes)
        tree_feature_rate: Fraction of features used by each tree (default: 0.5)
        n_class: Number of classes (default: 2)
        jointly_training: If True, jointly optimize routing and leaf parameters (default: False)
        epochs: Number of training epochs (default: 50)
        batch_size: Batch size for training (default: 64)
        lr: Learning rate (default: 0.001)
        weight_decay: L2 regularization (default: 5e-4)
        device: Device to use ('cuda' or 'cpu', default: auto-detect)
        random_state: Random seed (default: None)
    """

    def __init__(
        self,
        n_estimators=10,
        tree_depth=5,
        tree_feature_rate=0.5,
        n_class=2,
        jointly_training=False,
        epochs=50,
        batch_size=64,
        lr=0.001,
        weight_decay=5e-4,
        device=None,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.tree_depth_param = tree_depth  # Store original parameter for adaptive depth
        self.tree_depth = tree_depth  # Will be set adaptively in fit() if None
        self.tree_feature_rate = tree_feature_rate
        self.n_class = n_class
        self.jointly_training = jointly_training
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state

        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model_ = None
        self.n_features_ = None
        self.classes_ = None

    def _create_model(self, n_features):
        """Create a new NeuralDecisionForest model instance."""
        # Import from same directory
        from .NeuralDecisionForest import NeuralDecisionForest

        model = NeuralDecisionForest(
            n_features=n_features,
            n_tree=self.n_estimators,
            tree_depth=self.tree_depth,
            n_class=self.n_class,
            tree_feature_rate=self.tree_feature_rate,
            jointly_training=self.jointly_training
        )
        return model.to(self.device)

    def fit(self, X, y, sample_weight=None):
        """
        Fit the Neural Decision Forest model.

        Args:
            X: Training features [n_samples, n_features]
            y: Training labels [n_samples]
            sample_weight: Sample weights (not implemented yet)

        Returns:
            self
        """
        # Set random seed if provided
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Convert to numpy if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Store feature dimension and classes
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)

        # Adaptive tree depth: use depth 6 for <500 samples, depth 10 for ≥500 samples
        # This aligns with active learning scenarios where sample size ranges from ~100 to ~1200
        if self.tree_depth_param is None:
            n_samples = X.shape[0]
            self.tree_depth = 6 if n_samples < 500 else 10
        else:
            self.tree_depth = self.tree_depth_param

        # Create new model instance (fresh start for each fit)
        self.model_ = self._create_model(self.n_features_)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Setup optimizer and loss
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Optional: print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                # Uncomment for debugging:
                # print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: Features [n_samples, n_features]

        Returns:
            Class probabilities [n_samples, n_classes]
        """
        if self.model_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Convert to numpy if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Predict
        self.model_.eval()
        with torch.no_grad():
            probs = self.model_(X_tensor)
            probs = probs.cpu().numpy()

        return probs

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: Features [n_samples, n_features]

        Returns:
            Predicted class labels [n_samples]
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    # MolALKit interface methods
    def fit_molalkit(self, data, iteration: int = 0):
        """MolALKit interface for fitting."""
        X = data.X
        y = data.y
        assert y.ndim == 2
        assert y.shape[1] == 1
        y = y.ravel()
        return self.fit(X, y)

    def predict_value(self, data):
        """MolALKit interface for prediction (returns positive class probability)."""
        X = data.X
        probs = self.predict_proba(X)
        return probs[:, 1]  # Probability of positive class

    def predict_uncertainty(self, data):
        """
        MolALKit interface for uncertainty estimation.

        Uses variance-based uncertainty: uncertainty = (0.25 - var(p)) * 4
        This is the same metric used in BaseSklearnModel for classification.
        """
        X = data.X
        probs = self.predict_proba(X)
        return (0.25 - np.var(probs, axis=1)) * 4
