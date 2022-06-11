import math
import sys
import numpy as np

from pysmt.shortcuts import Symbol, And, Or, AtMostOne, get_model
from tqdm import tqdm


class BinaryTreeNode:
    def __init__(self, path_sequence, n, d, left_child=None, right_child=None):
        self.parent = None
        self.left_child = left_child
        self.right_child = right_child
        if self.left_child is not None:
            left_child.parent = self
        if self.right_child is not None:
            right_child.parent = self

        self.decision_roles = [[Symbol(f'{path_sequence}|D[{i},{j}]') for i in range(n - 1)] for j in range(d)]
        self.negative_leaf_node_role = Symbol(f'{path_sequence}|-')
        self.positive_leaf_node_role = Symbol(f'{path_sequence}|+')
        roles = [dr for drs in self.decision_roles for dr in drs] + [self.negative_leaf_node_role] + [self.positive_leaf_node_role]
        self.roles = roles

        # exact_role_clause_list = []
        # for true_role in roles:
        #     exact_role_clause_list.append(And([role if role is true_role else Not(role) for role in roles]))
        # no_role_clause = And([Not(role) for role in roles])
        # self.role_domain_clause = Or(Or(exact_role_clause_list), no_role_clause)

        self.role_domain_clause = AtMostOne(roles)


class BinaryTree:
    def build_node(self, path_sequence, n, d, h, progress=None):
        if h == 0:
            return None
        left_child = self.build_node(path_sequence + 'L', n, d, h - 1, progress)
        right_child = self.build_node(path_sequence + 'R', n, d, h - 1, progress)
        current = BinaryTreeNode(path_sequence, n, d, left_child, right_child)
        if progress is not None:
            progress.update()
        return current

    def build_root(self, n, d, h, silent=False):
        if silent:
            progress = None
        else:
            progress = tqdm(total=(2 ** (h - 1)) * 2 - 1, desc='Compiling role domain clause')

        root_node = self.build_node('', n, d, h, progress)

        if progress is not None:
            progress.close()

        return root_node

    def compile_role_domain_clauses(self, node, role_domain_clauses):
        if node.left_child is not None:
            self.compile_role_domain_clauses(node.left_child, role_domain_clauses)
        if node.right_child is not None:
            self.compile_role_domain_clauses(node.right_child, role_domain_clauses)
        role_domain_clauses.append(node.role_domain_clause)

    def compile_role_domain_clause(self):
        role_domain_clauses = []
        self.compile_role_domain_clauses(self.root, role_domain_clauses)
        return And(role_domain_clauses)

    def __init__(self, n, d, h):
        self.root = self.build_root(n, d, h)

        # The role domain clause, which asserts that every node is assigned
        # exactly one role, can be compiled now since it's constant w.r.t.
        # the data.
        self.role_domain_clause = self.compile_role_domain_clause()

    def compile_valid_point_propagation_clause(self, node, data_idx, data_labels, sorted_data_indices, model=None):
        cur_data_label = data_labels[data_idx]
        cur_sorted_data_indices = sorted_data_indices[:, data_idx]

        # Keep track of all of the clauses which specify a valid propagation
        # to a class-matching leaf node. The logical OR of all of these is
        # the final valid propagation clause.
        valid_point_propagation_clauses = []

        # Determine "clause" (actually just a single symbol) which
        # specifies whether the current node is a leaf node matching the
        # class of the data point. This is the "base case" for propagation to
        # a class-matching leaf node.
        if not cur_data_label:  # False label; i.e., "negative" class
            match_role = node.negative_leaf_node_role
        else:  # True label; i.e., "positive" class
            match_role = node.positive_leaf_node_role
        valid_point_propagation_clauses.append(match_role)

        # In the case where the current node isn't a leaf node matching
        # the data point's class, the data point needs to be propagated
        # down the tree, and two recursive checks need to be implemented
        # to see if the data point lands in a descendant leaf node matching
        # its class. However, if the current node doesn't have a left or right
        # child (at the bottom of the hypothetical tree), then we don't need
        # the corresponding propagation recursion at all.
        if node.left_child is not None:
            # Determine clause which specifies whether or not the data
            # point propagates leftward. This will be true if and only if
            # the current node has a decision role whose decision's
            # feature value is greater than or equal to that of the data
            # point.
            propagates_left_symbols = []
            for feature_idx in range(len(cur_sorted_data_indices)):
                cur_decision_roles = node.decision_roles[feature_idx]

                # Get the index of the first sorted feature value which is
                # greater than or equal to that of the current data point.
                first_ge_index = cur_sorted_data_indices[feature_idx]

                # For all sorted feature value indices greater than or equal
                # to first_ge_index, append them to propagates_left_symbols.
                # They're greater than or equal to the smallest value which
                # is greater than or equal to the data point's feature value.
                # Therefore, if any of them are used as the node's decision,
                # then the data point will propagate leftward.
                for compatible_idx in range(first_ge_index, len(data_labels) - 1):
                    propagates_left_symbols.append(cur_decision_roles[compatible_idx])
            propagates_left_clause = Or(propagates_left_symbols)

            # Determine clause which specifies whether or not the data
            # point recursively propagates to a leaf node which is a
            # left-descendant of the current node
            matches_left_descendant = self.compile_valid_point_propagation_clause(node.left_child, data_idx, data_labels, sorted_data_indices)

            # The data point ends up in a class-matching leaf node through
            # leftward propagation if and only if it indeed propagates
            # leftward and, following the remainder of the propagation path
            # recursively, ends up in a class-matching leaf node. This is the
            # logical AND of propagates_left_symbols and
            # matches_left_descendant
            valid_left = And(propagates_left_clause, matches_left_descendant)
            valid_point_propagation_clauses.append(valid_left)

        # Repeat all of this logic for rightward propagation
        if node.right_child is not None:
            propagates_right_symbols = []
            for feature_idx in range(len(cur_sorted_data_indices)):
                cur_decision_roles = node.decision_roles[feature_idx]
                first_ge_index = cur_sorted_data_indices[feature_idx]

                # Recall that first_ge_index is the index of the first sorted
                # feature value which is greater than or equal to that of the
                # data point. For right propagation, the decision value needs
                # to be less than that of the data point. So we use all of the
                # decision values which come BEFORE first_ge_index
                for compatible_idx in range(0, first_ge_index):
                    propagates_right_symbols.append(cur_decision_roles[compatible_idx])
            propagates_right_clause = Or(propagates_right_symbols)

            matches_right_descendant = self.compile_valid_point_propagation_clause(node.right_child, data_idx, data_labels, sorted_data_indices)
            valid_right = And(propagates_right_clause, matches_right_descendant)
            valid_point_propagation_clauses.append(valid_right)

        # The data point is propagated to a class-matching leaf node if and
        # only if the current node is a class-matching leaf node, or the
        # data point propagates leftward to a class-matching leaf node, or
        # the data point propagates rightward to a class-matching leaf node.
        # The latter two cases are to be ignored when the current node doesn't
        # have the corresponding children. This can be computed as the logical
        # OR of the relevant clauses.
        return Or(valid_point_propagation_clauses)

    def compile_valid_propagation_clause(self, data_labels, sorted_data_indices):
        # For every data point, check that it gets propagated to a
        # class-matching leaf node. This is the logical AND of a clause per
        # data point.
        valid_point_propagation_clauses = []
        progress = tqdm(total=len(data_labels), desc='Compiling valid propagation clause')
        for data_idx in range(len(data_labels)):
            valid_point_propagation_clauses.append(self.compile_valid_point_propagation_clause(self.root, data_idx, data_labels, sorted_data_indices))
            progress.update()
        progress.close()
        return And(valid_point_propagation_clauses)

    def compile_formula(self, data_labels, sorted_data_indices):
        # All nodes must have exactly one role. Use self.role_domain_clause.
        # AND
        # All data points must propagate to a class-matching leaf node
        valid_propagation_clause = self.compile_valid_propagation_clause(data_labels, sorted_data_indices)

        return And(self.role_domain_clause, valid_propagation_clause)


class TrackingTreeNode:
    def __init__(self, left_child=None, right_child=None):
        self.left_child = left_child
        self.right_child = right_child


class TrackingTree:
    def __init__(self, root):
        self.root = root


class DecisionTreeNode:
    def __init__(self, left_child=None, right_child=None, decision_idx=None, decision_value=None, leaf_class=None):
        assert (left_child is None and right_child is None and decision_idx is None and decision_value is None) or (leaf_class is None)
        assert (left_child is not None and right_child is not None and decision_idx is not None and decision_value is not None) or (leaf_class is not None)
        self.left_child = left_child
        self.right_child = right_child
        self.decision_idx = decision_idx
        self.decision_value = decision_value
        self.leaf_class = leaf_class

    def predict(self, x):
        if self.leaf_class is not None:
            return np.full(len(x), self.leaf_class, dtype=bool)
        if self.left_child is not None \
                and self.right_child is not None \
                and self.decision_idx is not None \
                and self.decision_value is not None:
            left_indices = x[:, self.decision_idx] <= self.decision_value
            right_indices = x[:, self.decision_idx] > self.decision_value

            y = np.empty(len(x), dtype=bool)

            left_x = x[left_indices]
            if len(left_x) > 0:
                left_y = self.left_child.predict(left_x)
                y[left_indices] = left_y
            right_x = x[right_indices]
            if len(right_x) > 0:
                right_y = self.right_child.predict(right_x)
                y[right_indices] = right_y

            return y

    def track(self, x):
        if self.left_child is not None \
                and self.right_child is not None \
                and self.decision_idx is not None \
                and self.decision_value is not None:
            left_indices = x[:, self.decision_idx] <= self.decision_value
            right_indices = x[:, self.decision_idx] > self.decision_value

            left_x = x[left_indices]
            left_tracking_tree_node = None
            if len(left_x) > 0:
                left_tracking_tree_node = self.left_child.track(left_x)
            right_x = x[right_indices]
            right_tracking_tree_node = None
            if len(right_x) > 0:
                right_tracking_tree_node = self.right_child.track(right_x)
            return TrackingTreeNode(left_child=left_tracking_tree_node, right_child=right_tracking_tree_node)
        else:
            return TrackingTreeNode()

    def track_prune(self, tracking_tree_node):
        # Leaf nodes don't have children to prune, and if we're calling this
        # method on a leaf node, then it's been tracked. So just return it
        # immediately
        if self.leaf_class is not None:
            return self

        # Otherwise, it's a tracked decision node. Recursively prune its
        # subtrees, one of which might be untracked, and both of which might
        # contain untracked sub-subtrees.
        pruned_left_subtree = None
        if tracking_tree_node.left_child is not None:
            pruned_left_subtree = self.left_child.track_prune(tracking_tree_node.left_child)
        pruned_right_subtree = None
        if tracking_tree_node.right_child is not None:
            pruned_right_subtree = self.right_child.track_prune(tracking_tree_node.right_child)

        # If both subtrees are pruned, then throw an error; this decision node
        # has been tracked, but somehow neither of its children survived
        # pruning
        assert pruned_left_subtree is not None or pruned_right_subtree is not None

        # If only one subtree exists after pruning, then this is a decision
        # node with only one child; collapse this node into the remaining
        # subtree. If both subtrees remain, then simply modify this node and
        # to record the new subtrees (which might have been collapsed
        # themselves) and return it.
        if pruned_left_subtree is not None and pruned_right_subtree is None:
            return pruned_left_subtree
        elif pruned_right_subtree is not None and pruned_right_subtree is None:
            return pruned_right_subtree
        else:
            self.left_child = pruned_left_subtree
            self.right_child = pruned_right_subtree
            return self


class DecisionTree:
    def __init__(self, root):
        self.root = root

    def predict(self, x):
        return self.root.predict(x)

    def track(self, x):
        return TrackingTree(self.root.track(x))

    def track_prune(self, x):
        tracking_tree = self.track(x)
        self.root = self.root.track_prune(tracking_tree.root)

    def from_symbol_node(symbol_node, sorted_data_values, model):
        value_dict = model.get_py_values(symbol_node.roles)
        if all(not value for value in value_dict.values()):
            return None

        negative_leaf_class = model.get_py_value(symbol_node.negative_leaf_node_role)
        positive_leaf_class = model.get_py_value(symbol_node.positive_leaf_node_role)
        if negative_leaf_class:
            leaf_class = False
        elif positive_leaf_class:
            leaf_class = True
        else:
            leaf_class = None

        if leaf_class is not None:
            return DecisionTreeNode(leaf_class=leaf_class)

        # If we make it this far, then the node is a decision node. However,
        # it might be at the bottom of the tree, in which case it doesn't
        # actually have children. This means that no training data gets
        # propagated to it; otherwise they wouldn't have passed the valid
        # propagation clause since they wouldn't have been propagated to a
        # leaf node. This whole unnecessary path will have to be pruned after
        # construction, but for now we can just prune the current node
        # (otherwise we have to represent a decision node with no children).
        # So we'll construct each child only if it exists in the symbol tree.
        left_child = None
        if symbol_node.left_child is not None:
            left_child = DecisionTree.from_symbol_node(symbol_node.left_child, sorted_data_values, model)
        right_child = None
        if symbol_node.right_child is not None:
            right_child = DecisionTree.from_symbol_node(symbol_node.right_child, sorted_data_values, model)

        # If one child is None (e.g., because it's marked as unused), then all
        # of the training data must have been propagated to the other child.
        # This means that this decision node is unnecessary. We can remove this
        # node from the tree by just returning the non-None child. If both
        # children are None, then this decision is unnecessary and there is no
        # sub-tree, so this node can just be set to None.
        if left_child is not None and right_child is None:
            return left_child
        elif right_child is not None and left_child is None:
            return right_child
        elif left_child is None and right_child is None:
            return None
        else:
            # If neither child is None, then this is a useful decision node.
            # We need to construct it and return it.

            # First, get the decision index and value
            for feature_idx, feature_symbols in enumerate(symbol_node.decision_roles):
                for data_idx, symbol in enumerate(feature_symbols):
                    if model.get_py_value(symbol):
                        # This is the node's decision role. Extract the value
                        # from sorted_data_value using the indices
                        decision_value = sorted_data_values[feature_idx][data_idx]
                        decision_idx = feature_idx
                        break

            # Now return the node
            return DecisionTreeNode(left_child=left_child, right_child=right_child, decision_idx=decision_idx, decision_value=decision_value)

    def from_symbol_tree(symbol_tree, data, sorted_data_values, model):
        # Crudely construct the tree. This will remove unused nodes as well as
        # decision nodes which don't propagate to leaf nodes. However, it
        # WON'T remove "empty" nodes, meaning nodes which don't have any
        # training data propagated to them. This is possible because there is
        # no SAT clause requiring every leaf node to be used.
        root = DecisionTree.from_symbol_node(symbol_tree.root, sorted_data_values, model)
        tree = DecisionTree(root)

        # Now we have to determine which nodes are empty for pruning.
        # tree.track_prune(data)

        return tree


def main():
    if len(sys.argv) != 3:
        raise ValueError("Bad calling syntax: {} vectors_fname labels_fname".format(sys.argv[0]))

    vectors_fname = sys.argv[1]
    labels_fname = sys.argv[2]

    vectors = np.loadtxt(vectors_fname, delimiter=',')
    labels = np.loadtxt(labels_fname, delimiter=',', dtype=bool)

    n, d = vectors.shape

    # When compiling the formula, for each feature dimension, for each data
    # point, the algorithm an index which specifies the first data point, when
    # sorted in ascending order by the feature values in the given dimension,
    # with a feature value in the given dimension which is greater than or
    # equal to that of the given data point. This can also be interpreted as
    # an integer specifying the number of feature values in the given dimension
    # smaller than that of the given data point.

    # For instance, suppose the N values of feature dimension 0 are organized
    # like so:

    # 7, 2, 9, 10, 1

    # Then the sorted list looks like this:
    # 1, 2, 7, 9, 10

    # Then for feature dimension 0, data point 0, the algorithm expects an
    # integer specifying the number of data points with feature values smaller
    # than it. Data point 0 has a feature value of 7, and there are two feature
    # values smaller than 7. So the correct index is 2.

    # However, this index does not always match the index of the data point
    # when sorted in ascending order. For instance, consider the case of
    # duplicates:

    # Unsorted: 7, 2, 7, 10, 1
    # Sorted: 1, 2, 7, 7, 9

    # In this case, consider data point 0. Its feature value is still greater
    # than exactly two other feature values. However, its index in the sorted
    # list MIGHT be 3, since there is a duplicate feature value.

    # We need such a number for each <data point, feature> pair. We can start
    # by sorting the data points in ascending order by each feature dimension
    # individually, keeping track of their original data indices as we do so.

    def sort_key(t):
        return t[1]
    sorted_data_lists = []
    for feature_idx in range(d):
        data_index_tuples = enumerate(vectors[:, feature_idx])
        sorted_data_lists.append(sorted(data_index_tuples, key=sort_key))

    # Next, for each <data point, feature> pair, we need to extract the correct
    # value from the sorted data lists and insert it at the correct place in
    # a 2D array
    sorted_data_indices = []
    for feature_idx in range(d):
        # Construct a dummy index list to be updated
        cur_sorted_data_indices = [0] * n
        # Update with the appropriate values at the appropriate locations
        cur_sorted_data_list = sorted_data_lists[feature_idx]
        prev_feature_value = None
        prev_first_ge_idx = None
        for sorted_data_idx, (original_data_idx, feature_value) in enumerate(cur_sorted_data_list):
            if prev_feature_value is not None and feature_value == prev_feature_value:
                # We have duplicate feature values in this dimension. Maintain
                # the previous first_ge_idx since the feature value hasn't
                # increased.
                cur_sorted_data_indices[original_data_idx] = prev_first_ge_idx
            else:
                # This is a new feature value, not a duplicate. It's greater
                # than all previous feature values due to the ascending sort.
                # Its sorted index, then, is equal to the number of elements
                # smaller than it. So use its sorted index as first_ge_idx
                cur_sorted_data_indices[original_data_idx] = sorted_data_idx
                prev_feature_value = feature_value
                prev_first_ge_idx = sorted_data_idx
        sorted_data_indices.append(cur_sorted_data_indices)
    sorted_data_indices = np.array(sorted_data_indices, dtype=int)

    # In summary, sorted_data_indices[i][j] is an integer specifying the number
    # of data points whose feature value at index j is smaller than that of
    # data point i's feature value at index j. Or equivalently, it specifies
    # the index of the first sorted data point whose feature j value is greater
    # than or equal to that of data point i, when sorted by feature j (and
    # this will always actually be an equality, because there will always be
    # a data point whose feature j value is equal to that of data point i,
    # namely itself. The "greater than or equal to" terminology is just easier
    # to understand in the context of the SAT reduction)

    for h in range(1, math.ceil(math.log(n, 2)) + 1):

        # Construct complete hypothetical tree
        tree = BinaryTree(n, d, h)
        formula = tree.compile_formula(labels, sorted_data_indices)

        model = get_model(formula)

        if model is not None:
            break

        print(f'No model found for h={h}')

    if model is None:
        print('No model found for any choice of h')
    else:
        print(f'Model found for h={h}')
        sorted_data_values = [[item[1] for item in sorted_data_list] for sorted_data_list in sorted_data_lists]
        dt = DecisionTree.from_symbol_tree(tree, vectors, sorted_data_values, model)
        predictions = dt.predict(vectors)
        correct = labels == predictions
        num_correct = correct.astype(int).sum()
        accuracy = float(num_correct) / len(labels)
        print(f'Accuracy: {accuracy}')
        # tree.compile_valid_point_propagation_clause(tree.root, 2, labels, sorted_data_indices, model=model)


if __name__ == '__main__':
    main()
