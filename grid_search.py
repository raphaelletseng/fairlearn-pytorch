import numpy as np
import pandas as pd
import logging
from time import time

from sklearn.dummy import DummyClassifier

from fairlearn.reductions import GridSearch
from fairlearn.reductions._moments import Moment, ClassificationMoment
from fairlearn.reductions._grid_search._grid_generator import _GridGenerator

logger = logging.getLogger(__name__)


TRADEOFF_OPTIMIZATION = "tradeoff_optimization"


class ModifiedGridSearch(GridSearch):
    r"""
    This is identical to the grid search from version
    0.6.1 of fairlearn. The only difference is that copy.deepcopy doesn't
    work for a skorch model with a Differential Privacy optimizer, with the following
    error message:
    TypeError: can't pickle torch._C.Generator objects

    So instead of copying estimators by using deepcopy, we instead pass a make_estimator_func
    which returns a new estimator.
    """
    def __init__(self,
                 estimator,
                 constraints,
                 make_estimator_func,
                 selection_rule=TRADEOFF_OPTIMIZATION,
                 constraint_weight=0.5,
                 grid_size=10,
                 grid_limit=2.0,
                 grid_offset=None,
                 grid=None,
                 sample_weight_name="sample_weight"):
        super().__init__(estimator, constraints, selection_rule,
            constraint_weight, grid_size, grid_limit, grid_offset,
            grid, sample_weight_name
        )
        self.make_estimator_func = make_estimator_func

    def fit(self, X, y, **kwargs):
        """Run the grid search.
        This will result in multiple copies of the
        estimator being made, and the :code:`fit(X)` method
        of each one called.
        :param X: The feature matrix
        :type X: numpy.ndarray or pandas.DataFrame
        :param y: The label vector
        :type y: numpy.ndarray, pandas.DataFrame, pandas.Series, or list
        :param sensitive_features: A (currently) required keyword argument listing the
            feature used by the constraints object
        :type sensitive_features: numpy.ndarray, pandas.DataFrame, pandas.Series, or list (for now)
        """
        self.predictors_ = []
        self.lambda_vecs_ = pd.DataFrame(dtype=np.float64)
        self.objectives_ = []
        self.gammas_ = pd.DataFrame(dtype=np.float64)
        self.oracle_execution_times_ = []

        if isinstance(self.constraints, ClassificationMoment):
            logger.debug("Classification problem detected")
            is_classification_reduction = True
        else:
            logger.debug("Regression problem detected")
            is_classification_reduction = False

        # Prep the parity constraints and objective
        logger.debug("Preparing constraints and objective")
        self.constraints.load_data(X, y, **kwargs)
        objective = self.constraints.default_objective()
        objective.load_data(X, y, **kwargs)

        # Basis information
        pos_basis = self.constraints.pos_basis
        neg_basis = self.constraints.neg_basis
        neg_allowed = self.constraints.neg_basis_present
        objective_in_the_span = (self.constraints.default_objective_lambda_vec is not None)

        if self.grid is None:
            logger.debug("Creating grid of size %i", self.grid_size)
            grid = _GridGenerator(self.grid_size,
                                  self.grid_limit,
                                  pos_basis,
                                  neg_basis,
                                  neg_allowed,
                                  objective_in_the_span,
                                  self.grid_offset).grid
        else:
            logger.debug("Using supplied grid")
            grid = self.grid

        # Fit the estimates
        logger.debug("Setup complete. Starting grid search")
        for i in grid.columns:
            lambda_vec = grid[i]
            logger.debug("Obtaining weights")
            weights = self.constraints.signed_weights(lambda_vec)
            if not objective_in_the_span:
                weights = weights + objective.signed_weights()

            if is_classification_reduction:
                logger.debug("Applying relabelling for classification problem")
                y_reduction = 1 * (weights > 0)
                weights = weights.abs()
            else:
                y_reduction = self.constraints._y_as_series

            y_reduction_unique = np.unique(y_reduction)
            if len(y_reduction_unique) == 1:
                logger.debug("y_reduction had single value. Using DummyClassifier")
                current_estimator = DummyClassifier(strategy='constant',
                                                    constant=y_reduction_unique[0])
            else:
                logger.debug("Using underlying estimator")
                current_estimator = self.make_estimator_func()

            oracle_call_start_time = time()
            current_estimator.fit(X, y_reduction, **{self.sample_weight_name: weights})
            oracle_call_execution_time = time() - oracle_call_start_time
            logger.debug("Call to estimator complete")

            def predict_fct(X): return current_estimator.predict(X)
            self.predictors_.append(current_estimator)
            self.lambda_vecs_[i] = lambda_vec
            self.objectives_.append(objective.gamma(predict_fct)[0])
            self.gammas_[i] = self.constraints.gamma(predict_fct)
            self.oracle_execution_times_.append(oracle_call_execution_time)

        logger.debug("Selecting best_result")
        if self.selection_rule == TRADEOFF_OPTIMIZATION:
            def loss_fct(i):
                return self.objective_weight * self.objectives_[i] + \
                    self.constraint_weight * self.gammas_[i].max()
            losses = [loss_fct(i) for i in range(len(self.objectives_))]
            self.best_idx_ = losses.index(min(losses))
        else:
            raise RuntimeError("Unsupported selection rule")

        return
