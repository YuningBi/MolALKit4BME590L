"""
Modular Neural Decision Forest - Generic version without feature layers.

This version accepts pre-computed feature vectors (from any source like
DeepChem, sklearn, or other feature extractors) and focuses on the core
decision forest learning.

Useful for:
- Tox21 and other molecular datasets (with pre-computed fingerprints)
- Integration into existing ML pipelines
- Comparison with other methods (RF, XGB, etc.)
- Transfer learning scenarios

Author: Based on Neural Decision Forests by Jingxil
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F


class Tree(nn.Module):
    """
    Decision tree with soft routing and learnable leaf distributions.

    Args:
        depth: Tree depth (number of levels)
        n_in_feature: Input feature dimension
        used_feature_rate: Fraction of features to use in this tree (for regularization)
        n_class: Number of classes
        jointly_training: If True, leaf distributions are trainable parameters
    """
    def __init__(self, depth, n_in_feature, used_feature_rate, n_class, jointly_training=True):
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.n_class = n_class
        self.jointly_training = jointly_training

        # Feature masking: each tree uses only a subset of features
        n_used_feature = int(n_in_feature * used_feature_rate)
        onehot = np.eye(n_in_feature)
        using_idx = np.random.choice(np.arange(n_in_feature), n_used_feature, replace=False)
        self.feature_mask = onehot[using_idx].T
        self.feature_mask = Parameter(
            torch.from_numpy(self.feature_mask).type(torch.FloatTensor),
            requires_grad=False
        )

        # Leaf label distribution (probability of each class at each leaf)
        if jointly_training:
            # Trainable: initialized randomly
            self.pi = np.random.rand(self.n_leaf, n_class)
            self.pi = Parameter(
                torch.from_numpy(self.pi).type(torch.FloatTensor),
                requires_grad=True
            )
        else:
            # Fixed: initialized uniformly
            self.pi = np.ones((self.n_leaf, n_class)) / n_class
            self.pi = Parameter(
                torch.from_numpy(self.pi).type(torch.FloatTensor),
                requires_grad=False
            )

        # Decision node: soft routing via sigmoid
        self.decision = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(n_used_feature, self.n_leaf)),
            ('sigmoid', nn.Sigmoid()),
        ]))

    def forward(self, x):
        """
        Compute routing probabilities to tree leaves.

        Args:
            x: Input features [batch_size, n_features]

        Returns:
            Route probability to each leaf [batch_size, n_leaf]
        """
        if x.is_cuda and not self.feature_mask.is_cuda:
            self.feature_mask = self.feature_mask.cuda()

        # Apply feature mask: select subset of features
        feats = torch.mm(x, self.feature_mask)  # [batch_size, n_used_feature]

        # Soft routing decisions
        decision = self.decision(feats)  # [batch_size, n_leaf]

        # Convert to probabilistic routing: each node outputs go/no-go probability
        decision = torch.unsqueeze(decision, dim=2)
        decision_comp = 1 - decision
        decision = torch.cat((decision, decision_comp), dim=2)  # [batch_size, n_leaf, 2]

        # Compute cumulative routing probability through the tree
        batch_size = x.size()[0]
        _mu = Variable(x.data.new(batch_size, 1, 1).fill_(1.))
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _decision = decision[:, begin_idx:end_idx, :]  # [batch_size, 2**n_layer, 2]
            _mu = _mu * _decision  # [batch_size, 2**n_layer, 2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer + 1)

        mu = _mu.view(batch_size, self.n_leaf)
        return mu

    def get_pi(self):
        """Get leaf class distribution (apply softmax if jointly training)."""
        if self.jointly_training:
            return F.softmax(self.pi, dim=-1)
        else:
            return self.pi

    def cal_prob(self, mu, pi):
        """
        Compute class probability from routing and leaf distributions.

        Args:
            mu: Routing probabilities [batch_size, n_leaf]
            pi: Leaf distributions [n_leaf, n_class]

        Returns:
            Class probability [batch_size, n_class]
        """
        p = torch.mm(mu, pi)
        return p

    def update_pi(self, new_pi):
        """Update leaf distributions (for two-stage training)."""
        self.pi.data = new_pi


class Forest(nn.Module):
    """
    Ensemble of decision trees.

    Combines predictions from multiple trees by averaging their probability outputs.
    """
    def __init__(self, n_tree, tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree = n_tree
        for _ in range(n_tree):
            tree = Tree(tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training)
            self.trees.append(tree)

    def forward(self, x):
        """
        Forward pass through all trees and average predictions.

        Args:
            x: Input features [batch_size, n_features]

        Returns:
            Averaged class probability [batch_size, n_class]
        """
        probs = []
        for tree in self.trees:
            mu = tree(x)
            p = tree.cal_prob(mu, tree.get_pi())
            probs.append(p.unsqueeze(2))
        probs = torch.cat(probs, dim=2)
        prob = torch.sum(probs, dim=2) / self.n_tree
        return prob


class NeuralDecisionForest(nn.Module):
    """
    Modular Neural Decision Forest - expects pre-computed feature vectors.

    This is a lightweight wrapper around Forest that directly accepts
    feature vectors (no feature extraction layer). Ideal for:
    - Pre-computed features (e.g., molecular fingerprints from DeepChem)
    - Integration into existing ML pipelines
    - Comparison with other methods

    Args:
        n_features: Dimension of input feature vectors
        n_tree: Number of trees in the forest
        tree_depth: Depth of each tree
        n_class: Number of classes
        tree_feature_rate: Fraction of features used by each tree (default: 0.5)
        jointly_training: If True, jointly optimize routing and leaf parameters (default: False)
    """
    def __init__(
        self,
        n_features,
        n_tree=10,
        tree_depth=5,
        n_class=2,
        tree_feature_rate=0.5,
        jointly_training=False
    ):
        super(NeuralDecisionForest, self).__init__()
        self.n_features = n_features
        self.forest = Forest(
            n_tree=n_tree,
            tree_depth=tree_depth,
            n_in_feature=n_features,
            tree_feature_rate=tree_feature_rate,
            n_class=n_class,
            jointly_training=jointly_training
        )

    def forward(self, x):
        """
        Args:
            x: Feature vectors [batch_size, n_features]

        Returns:
            Class probabilities [batch_size, n_class]
        """
        return self.forest(x)
