"""Classes for Transforming and Building Data."""
import functools
import itertools

import numpy as np

from .consumed import interaction_consumed, update_consumed
from .data_info import DataInfo, store_old_info
from .transformed import LazyTransformedSet, TransformedEvalSet, TransformedSet
from ..feature.column_mapping import col_name2index
from ..feature.multi_sparse import (
    get_multi_sparse_info,
    multi_sparse_col_map,
    recover_sparse_cols,
)
from ..feature.sparse import (
    get_id_indices,
    get_oov_pos,
    merge_offset,
    merge_sparse_col,
    merge_sparse_indices,
)
from ..feature.unique import construct_unique_feat
from ..feature.update import (
    update_id_unique,
    update_multi_sparse_unique,
    update_sparse_unique,
    update_unique_feats,
)


class _Dataset(object):
    """Base class for loading dataset."""

    user_unique_vals = None
    item_unique_vals = None
    train_called = False

    @staticmethod
    def _check_col_names(data, is_train):
        if not np.all(["user" == data.columns[0], "item" == data.columns[1]]):
            raise ValueError("'user', 'item' must be the first two columns of the data")
        if is_train:
            assert "label" in data.columns, "train data should contain label column"

    @classmethod
    def _check_subclass(cls):
        if not issubclass(cls, _Dataset):
            raise NameError("Please use 'DatasetPure' or 'DatasetFeat' to call method")

    @staticmethod
    def shuffle_data(data, seed):
        """Shuffle data randomly.

        Parameters
        ----------
        data : pandas.DataFrame
            Data to shuffle.
        seed : int
            Random seed.

        Returns
        -------
        pandas.DataFrame
            Shuffled data.
        """
        data = data.sample(frac=1, random_state=seed)
        return data.reset_index(drop=True)

    @classmethod
    def _transform_test_factory(cls, test_data, shuffle, seed, data_info=None):
        if not cls.train_called:
            raise RuntimeError(
                "Must first build trainset before building evalset or testset"
            )
        cls._check_subclass()
        cls._check_col_names(test_data, is_train=False)
        if shuffle:
            test_data = cls.shuffle_data(test_data, seed)

        if cls.__name__ == "DatasetPure":
            return _build_transformed_set(
                test_data,
                cls.user_unique_vals,
                cls.item_unique_vals,
                is_train=False,
                is_ordered=False,
            )
        else:
            return _build_transformed_set_feat(
                test_data,
                cls.user_unique_vals,
                cls.item_unique_vals,
                is_train=False,
                is_ordered=False,
                data_info=data_info,
            )

    @classmethod
    def build_evalset(cls, eval_data, shuffle=False, seed=42):
        """Build transformed eval data from original data.

        .. versionchanged:: 1.0.0
           Data construction in :ref:`Model Retrain <retrain_data>` has been moved
           to :meth:`merge_evalset`

        Parameters
        ----------
        eval_data : pandas.DataFrame
            Data must contain at least two columns, i.e. `user`, `item`.
        shuffle : bool, default: False
            Whether to fully shuffle data.
        seed: int, default: 42
            Random seed.

        Returns
        -------
        :class:`~libreco.data.TransformedEvalSet`
            Transformed Data object used for evaluating.
        """
        return cls._transform_test_factory(eval_data, shuffle, seed)

    @classmethod
    def build_testset(cls, test_data, shuffle=False, seed=42):
        """Build transformed test data from original data.

        .. versionchanged:: 1.0.0
           Data construction in :ref:`Model Retrain <retrain_data>` has been moved
           to :meth:`merge_testset`

        Parameters
        ----------
        test_data : pandas.DataFrame
            Data must contain at least two columns, i.e. `user`, `item`.
        shuffle : bool, default: False
            Whether to fully shuffle data.
        seed: int, default: 42
            Random seed.

        Returns
        -------
        :class:`~libreco.data.TransformedEvalSet`
            Transformed Data object used for testing.
        """
        return cls._transform_test_factory(test_data, shuffle, seed)

    @classmethod
    def merge_evalset(cls, eval_data, data_info, shuffle=False, seed=42):
        """Build transformed data by merging new train data with old data.

        .. versionadded:: 1.0.0

        Parameters
        ----------
        eval_data : pandas.DataFrame
            Data must contain at least two columns, i.e. `user`, `item`.
        data_info : DataInfo
            Object that contains past data information.
        shuffle : bool, default: False
            Whether to fully shuffle data.
        seed: int, default: 42
            Random seed.

        Returns
        -------
        :class:`~libreco.data.TransformedEvalSet`
            Transformed Data object used for testing.
        """
        return cls._transform_test_factory(eval_data, shuffle, seed, data_info)

    @classmethod
    def merge_testset(cls, test_data, data_info, shuffle=False, seed=42):
        """Build transformed data by merging new train data with old data.

        .. versionadded:: 1.0.0

        Parameters
        ----------
        test_data : pandas.DataFrame
            Data must contain at least two columns, i.e. `user`, `item`.
        data_info : DataInfo
            Object that contains past data information.
        shuffle : bool, default: False
            Whether to fully shuffle data.
        seed: int, default: 42
            Random seed.

        Returns
        -------
        :class:`~libreco.data.TransformedEvalSet`
            Transformed Data object used for testing.
        """
        return cls._transform_test_factory(test_data, shuffle, seed, data_info)


class DatasetPure(_Dataset):
    """Dataset class used for building pure collaborative filtering data.

    Examples
    --------
    >>> from libreco.data import DatasetPure
    >>> train_data, data_info = DatasetPure.build_trainset(train_data)
    >>> eval_data = DatasetPure.build_evalset(eval_data)
    >>> test_data = DatasetPure.build_testset(test_data)
    """

    @classmethod
    def build_trainset(cls, train_data, shuffle=False, seed=42):
        """Build transformed train data and data_info from original data.

        .. versionchanged:: 1.0.0
           Data construction in :ref:`Model Retrain <retrain_data>` has been moved
           to :meth:`merge_trainset`

        Parameters
        ----------
        train_data : pandas.DataFrame
            Data must contain at least three columns, i.e. ``user``, ``item``, ``label``.
        shuffle : bool, default: False
            Whether to fully shuffle data.

            .. Warning::
                If your data is order or time dependent, it is not recommended to shuffle data.

        seed: int, default: 42
            Random seed.

        Returns
        -------
        trainset : :class:`~libreco.data.TransformedSet`
            Transformed Data object used for training.
        data_info : :class:`~libreco.data.DataInfo`
            Object that contains some useful information.
        """
        cls._check_subclass()
        cls._check_col_names(train_data, is_train=True)
        cls.user_unique_vals = np.sort(train_data["user"].unique())
        cls.item_unique_vals = np.sort(train_data["item"].unique())
        if shuffle:
            train_data = cls.shuffle_data(train_data, seed)

        train_transformed, user_indices, item_indices = _build_transformed_set(
            train_data,
            cls.user_unique_vals,
            cls.item_unique_vals,
            is_train=True,
            is_ordered=True,
        )
        user_consumed, item_consumed = interaction_consumed(user_indices, item_indices)
        data_info = DataInfo(
            interaction_data=train_data[["user", "item", "label"]],
            user_consumed=user_consumed,
            item_consumed=item_consumed,
            user_unique_vals=cls.user_unique_vals,
            item_unique_vals=cls.item_unique_vals,
            seed=seed,
        )
        cls.train_called = True
        return train_transformed, data_info

    @classmethod
    def merge_trainset(
        cls, train_data, data_info, merge_behavior=True, shuffle=False, seed=42
    ):
        """Build transformed data by merging new train data with old data.

        .. versionadded:: 1.0.0

        .. versionchanged:: 1.1.0
           Applying a more functional approach. A new ``data_info`` will be constructed
           and returned, and the passed old ``data_info`` should be discarded.

        Parameters
        ----------
        train_data : pandas.DataFrame
            Data must contain at least three columns, i.e. ``user``, ``item``, ``label``.
        data_info : DataInfo
            Object that contains past data information.
        merge_behavior : bool, default: True
            Whether to merge the user behavior in old and new data.
        shuffle : bool, default: False
            Whether to fully shuffle data.
        seed: int, default: 42
            Random seed.

        Returns
        -------
        new_trainset : :class:`~libreco.data.TransformedSet`
            New transformed Data object used for training.
        new_data_info : :class:`~libreco.data.DataInfo`
            New ``data_info`` that contains some useful information.
        """
        assert isinstance(data_info, DataInfo), "Invalid passed `data_info`."
        cls._check_col_names(train_data, is_train=True)
        cls.user_unique_vals, cls.item_unique_vals = update_id_unique(
            train_data, data_info
        )
        if shuffle:
            train_data = cls.shuffle_data(train_data, seed)

        merge_transformed, user_indices, item_indices = _build_transformed_set(
            train_data,
            cls.user_unique_vals,
            cls.item_unique_vals,
            is_train=True,
            is_ordered=False,
        )
        user_consumed, item_consumed = update_consumed(
            user_indices,
            item_indices,
            len(cls.user_unique_vals),
            len(cls.item_unique_vals),
            data_info,
            merge_behavior,
        )

        new_data_info = DataInfo(
            interaction_data=train_data[["user", "item", "label"]],
            user_consumed=user_consumed,
            item_consumed=item_consumed,
            user_unique_vals=cls.user_unique_vals,
            item_unique_vals=cls.item_unique_vals,
            seed=seed,
        )
        new_data_info.old_info = store_old_info(data_info)
        cls.train_called = True
        return merge_transformed, new_data_info


class DatasetFeat(_Dataset):
    """Dataset class used for building data contains features.

    Examples
    --------
    >>> from libreco.data import DatasetFeat
    >>> train_data, data_info = DatasetFeat.build_trainset(train_data)
    >>> eval_data = DatasetFeat.build_evalset(eval_data)
    >>> test_data = DatasetFeat.build_testset(test_data)
    """

    sparse_unique_vals = None
    multi_sparse_unique_vals = None
    sparse_col = None
    multi_sparse_col = None
    dense_col = None

    @classmethod
    def _set_feature_col(cls, sparse_col, dense_col, multi_sparse_col):
        cls.sparse_col = sparse_col or None
        cls.dense_col = dense_col or None
        if multi_sparse_col:
            if not all(isinstance(field, list) for field in multi_sparse_col):
                cls.multi_sparse_col = [multi_sparse_col]
            else:
                cls.multi_sparse_col = multi_sparse_col
        else:
            cls.multi_sparse_col = None

    @classmethod
    def _check_feature_cols(cls, user_col, item_col):
        all_sparse_col = (
            merge_sparse_col(cls.sparse_col, cls.multi_sparse_col)
            if cls.multi_sparse_col is not None
            else cls.sparse_col
        )
        sparse_cols = all_sparse_col or []
        dense_cols = cls.dense_col or []
        user_cols = user_col or []
        item_cols = item_col or []
        if len(sparse_cols) + len(dense_cols) != len(user_cols) + len(item_cols):
            len_str = "len(sparse_cols) + len(dense_cols) == len(user_cols) + len(item_cols)"  # fmt: skip
            raise ValueError(
                f"Please make sure length of columns match, i.e. `{len_str}`, got "
                f"sparse columns: {sparse_cols}, "
                f"dense columns: {dense_cols}, "
                f"user columns: {user_cols}, "
                f"item columns: {item_cols}"
            )
        columns1, columns2 = sparse_cols + dense_cols, user_cols + item_cols
        mis_match_cols = np.setxor1d(columns1, columns2)
        if len(mis_match_cols) > 0:
            raise ValueError(
                f"Got inconsistent columns: {mis_match_cols}, please check the column names"
            )

    @classmethod  # TODO: pseudo pure
    def build_trainset(
        cls,
        train_data,
        user_col=None,
        item_col=None,
        sparse_col=None,
        dense_col=None,
        multi_sparse_col=None,
        unique_feat=False,
        pad_val="missing",
        shuffle=False,
        seed=42,
    ):
        """Build transformed feat train data and data_info from original data.

        .. versionchanged:: 1.0.0
           Data construction in :ref:`Model Retrain <retrain_data>` has been moved
           to :meth:`merge_trainset`

        Parameters
        ----------
        train_data : pandas.DataFrame
            Data must contain at least three columns, i.e. ``user``, ``item``, ``label``.
        user_col : list of str or None, default: None
            List of user feature column names.
        item_col : list of str or None, default: None
            List of item feature column names.
        sparse_col : list of str or None, default: None
            List of sparse feature columns names.
        multi_sparse_col : nested lists of str or None, default: None
            Nested lists of multi_sparse feature columns names.
            For example, ``[["a", "b", "c"], ["d", "e"]]``
        dense_col : list of str or None, default: None
            List of dense feature column names.
        unique_feat : bool, default: False
            Whether the features of users and items are unique in train data.
        pad_val : int or str or list, default: "missing"
            Padding value in multi_sparse columns to ensure same length of all samples.

            .. Warning::
                If the ``pad_val`` is a single value, it will be used in all ``multi_sparse`` columns.
                So if you want to use different ``pad_val`` for different ``multi_sparse`` columns,
                the ``pad_val`` should be a list.

        shuffle : bool, default: False
            Whether to fully shuffle data.

            .. Warning::
                If your data is order or time dependent, it is not recommended to shuffle data.

        seed: int, default: 42
            Random seed.

        Returns
        -------
        trainset : :class:`~libreco.data.TransformedSet`
            Transformed Data object used for training.
        data_info : :class:`~libreco.data.DataInfo`
            Object that contains some useful information.

        Raises
        ------
        ValueError
            If the feature columns specified by the user are inconsistent.
        """
        cls._check_subclass()
        cls._check_col_names(train_data, is_train=True)
        cls._set_feature_col(sparse_col, dense_col, multi_sparse_col)
        cls._check_feature_cols(user_col, item_col)
        cls.user_unique_vals = np.sort(train_data["user"].unique())
        cls.item_unique_vals = np.sort(train_data["item"].unique())
        cls.sparse_unique_vals = _get_sparse_unique_vals(cls.sparse_col, train_data)
        cls.multi_sparse_unique_vals, pad_val_dict = _get_multi_sparse_unique_vals(
            cls.multi_sparse_col, train_data, pad_val
        )
        if shuffle:
            train_data = cls.shuffle_data(train_data, seed)

        (
            train_transformed,
            user_indices,
            item_indices,
            train_sparse_indices,
            train_dense_values,
        ) = _build_transformed_set_feat(
            train_data,
            cls.user_unique_vals,
            cls.item_unique_vals,
            is_train=True,
            is_ordered=True,
        )

        all_sparse_col = (
            merge_sparse_col(cls.sparse_col, cls.multi_sparse_col)
            if cls.multi_sparse_col
            else sparse_col
        )
        col_name_mapping = col_name2index(
            user_col, item_col, all_sparse_col, cls.dense_col
        )
        (
            user_sparse_unique,
            user_dense_unique,
            item_sparse_unique,
            item_dense_unique,
        ) = construct_unique_feat(
            user_indices,
            item_indices,
            train_sparse_indices,
            train_dense_values,
            col_name_mapping,
            unique_feat,
        )

        sparse_offset = merge_offset(
            cls.sparse_col,
            cls.multi_sparse_col,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
        )
        sparse_oov = get_oov_pos(
            cls.sparse_col,
            cls.multi_sparse_col,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
        )
        multi_sparse_info = get_multi_sparse_info(
            all_sparse_col,
            cls.sparse_col,
            cls.multi_sparse_col,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
            pad_val_dict,
        )
        if cls.multi_sparse_col:
            col_name_mapping["multi_sparse"] = multi_sparse_col_map(multi_sparse_col)

        interaction_data = train_data[["user", "item", "label"]]
        user_consumed, item_consumed = interaction_consumed(user_indices, item_indices)
        data_info = DataInfo(
            col_name_mapping,
            interaction_data,
            user_sparse_unique,
            user_dense_unique,
            item_sparse_unique,
            item_dense_unique,
            user_consumed,
            item_consumed,
            cls.user_unique_vals,
            cls.item_unique_vals,
            cls.sparse_unique_vals,
            sparse_offset,
            sparse_oov,
            cls.multi_sparse_unique_vals,
            multi_sparse_info,
            seed,
        )
        cls.train_called = True
        return train_transformed, data_info

    @classmethod
    def build_trainset_lazy(
        cls,
        interactions,
        user_features=None,
        item_features=None,
        user_col=None,
        item_col=None,
        sparse_col=None,
        dense_col=None,
        multi_sparse_col=None,
        pad_val="missing",
        shuffle=False,
        seed=42,
    ):
        """Build transformed train data with lazy feature loading for memory efficiency.

        Instead of expanding features for every interaction row, this method stores
        feature DataFrames separately and joins them on-the-fly during batch processing.
        This significantly reduces memory usage for large datasets.

        Parameters
        ----------
        interactions : pandas.DataFrame
            Interaction data containing ``user``, ``item``, ``label`` columns.
        user_features : pandas.DataFrame or None, default: None
            User features DataFrame containing ``user`` column and feature columns.
            Features will be joined with interactions during batch processing.
        item_features : pandas.DataFrame or None, default: None
            Item features DataFrame containing ``item`` column and feature columns.
            Features will be joined with interactions during batch processing.
        user_col : list of str or None, default: None
            List of user feature column names (must exist in user_features).
        item_col : list of str or None, default: None
            List of item feature column names (must exist in item_features).
        sparse_col : list of str or None, default: None
            List of sparse feature columns names.
        multi_sparse_col : nested lists of str or None, default: None
            Nested lists of multi_sparse feature columns names.
        dense_col : list of str or None, default: None
            List of dense feature column names.
        pad_val : int or str or list, default: "missing"
            Padding value in multi_sparse columns.
        shuffle : bool, default: False
            Whether to fully shuffle interaction data.
        seed: int, default: 42
            Random seed.

        Returns
        -------
        trainset : :class:`~libreco.data.LazyTransformedSet`
            Transformed Data object with lazy feature loading.
        data_info : :class:`~libreco.data.DataInfo`
            Object that contains some useful information.

        Examples
        --------
        >>> from libreco.data import DatasetFeat
        >>> # Separate interactions from features
        >>> interactions = train_df[["user", "item", "label"]]
        >>> user_features = train_df[["user", "age", "gender"]].drop_duplicates("user")
        >>> item_features = train_df[["item", "category"]].drop_duplicates("item")
        >>> train_data, data_info = DatasetFeat.build_trainset_lazy(
        ...     interactions=interactions,
        ...     user_features=user_features,
        ...     item_features=item_features,
        ...     user_col=["age", "gender"],
        ...     item_col=["category"],
        ...     sparse_col=["age", "gender", "category"],
        ... )
        """
        cls._check_subclass()
        cls._check_col_names(interactions, is_train=True)
        cls._set_feature_col(sparse_col, dense_col, multi_sparse_col)
        cls._check_feature_cols(user_col, item_col)

        # Extract unique values from interactions
        cls.user_unique_vals = np.sort(interactions["user"].unique())
        cls.item_unique_vals = np.sort(interactions["item"].unique())

        # Get sparse unique values from feature DataFrames
        cls.sparse_unique_vals = _get_sparse_unique_vals_lazy(
            cls.sparse_col, user_features, item_features
        )
        cls.multi_sparse_unique_vals, pad_val_dict = _get_multi_sparse_unique_vals_lazy(
            cls.multi_sparse_col, user_features, item_features, pad_val
        )

        if shuffle:
            interactions = cls.shuffle_data(interactions, seed)

        # Build basic transformed set (just user, item, labels)
        user_indices, item_indices = get_id_indices(
            interactions,
            cls.user_unique_vals,
            cls.item_unique_vals,
            is_train=True,
            is_ordered=True,
        )
        labels = interactions["label"].to_numpy(dtype=np.float32)

        # Prepare feature column lists
        all_sparse_col = (
            merge_sparse_col(cls.sparse_col, cls.multi_sparse_col)
            if cls.multi_sparse_col
            else sparse_col
        )

        # Build column name mapping
        col_name_mapping = col_name2index(
            user_col, item_col, all_sparse_col, cls.dense_col
        )

        # Identify which sparse/dense columns belong to user vs item
        user_sparse_col_names = []
        user_dense_col_names = []
        item_sparse_col_names = []
        item_dense_col_names = []

        if user_col:
            for col in user_col:
                if all_sparse_col and col in all_sparse_col:
                    user_sparse_col_names.append(col)
                elif cls.dense_col and col in cls.dense_col:
                    user_dense_col_names.append(col)

        if item_col:
            for col in item_col:
                if all_sparse_col and col in all_sparse_col:
                    item_sparse_col_names.append(col)
                elif cls.dense_col and col in cls.dense_col:
                    item_dense_col_names.append(col)

        # Build unique feature arrays from feature DataFrames
        (
            user_sparse_unique,
            user_dense_unique,
            item_sparse_unique,
            item_dense_unique,
        ) = _construct_unique_feat_lazy(
            cls.user_unique_vals,
            cls.item_unique_vals,
            user_features,
            item_features,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
            col_name_mapping,
            user_sparse_col_names,
            user_dense_col_names,
            item_sparse_col_names,
            item_dense_col_names,
        )

        # Build offset and oov info
        sparse_offset = merge_offset(
            cls.sparse_col,
            cls.multi_sparse_col,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
        )
        sparse_oov = get_oov_pos(
            cls.sparse_col,
            cls.multi_sparse_col,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
        )
        multi_sparse_info = get_multi_sparse_info(
            all_sparse_col,
            cls.sparse_col,
            cls.multi_sparse_col,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
            pad_val_dict if pad_val_dict else dict(),
        )
        if cls.multi_sparse_col:
            col_name_mapping["multi_sparse"] = multi_sparse_col_map(multi_sparse_col)

        # Create LazyTransformedSet
        train_transformed = LazyTransformedSet(
            user_indices=user_indices,
            item_indices=item_indices,
            labels=labels,
            user_features_df=user_features,
            item_features_df=item_features,
            sparse_col=cls.sparse_col,
            dense_col=cls.dense_col,
            multi_sparse_col=cls.multi_sparse_col,
            user_sparse_col=user_sparse_col_names if user_sparse_col_names else None,
            user_dense_col=user_dense_col_names if user_dense_col_names else None,
            item_sparse_col=item_sparse_col_names if item_sparse_col_names else None,
            item_dense_col=item_dense_col_names if item_dense_col_names else None,
        )

        interaction_data = interactions[["user", "item", "label"]]
        user_consumed, item_consumed = interaction_consumed(user_indices, item_indices)
        data_info = DataInfo(
            col_name_mapping,
            interaction_data,
            user_sparse_unique,
            user_dense_unique,
            item_sparse_unique,
            item_dense_unique,
            user_consumed,
            item_consumed,
            cls.user_unique_vals,
            cls.item_unique_vals,
            cls.sparse_unique_vals,
            sparse_offset,
            sparse_oov,
            cls.multi_sparse_unique_vals,
            multi_sparse_info,
            seed,
        )
        # Store feature DataFrames and column info in data_info for lazy loading
        data_info.lazy_user_features_df = user_features
        data_info.lazy_item_features_df = item_features
        data_info.lazy_mode = True

        cls.train_called = True
        return train_transformed, data_info

    @classmethod
    def merge_trainset(
        cls, train_data, data_info, merge_behavior=True, shuffle=False, seed=42
    ):
        """Build transformed data by merging new train data with old data.

        .. versionadded:: 1.0.0

        .. versionchanged:: 1.1.0
           Applying a more functional approach. A new ``data_info`` will be constructed
           and returned, and the passed old ``data_info`` should be discarded.

        Parameters
        ----------
        train_data : pandas.DataFrame
            Data must contain at least three columns, i.e. ``user``, ``item``, ``label``.
        data_info : DataInfo
            Object that contains past data information.
        merge_behavior : bool, default: True
            Whether to merge the user behavior in old and new data.
        shuffle : bool, default: False
            Whether to fully shuffle data.
        seed: int, default: 42
            Random seed.

        Returns
        -------
        new_trainset : :class:`~libreco.data.TransformedSet`
            New transformed Data object used for training.
        new_data_info : :class:`~libreco.data.DataInfo`
            New ``data_info`` that contains some useful information.
        """
        assert isinstance(data_info, DataInfo), "Invalid passed `data_info`."
        cls._check_col_names(train_data, is_train=True)
        cls.user_unique_vals, cls.item_unique_vals = update_id_unique(
            train_data, data_info
        )
        cls.sparse_unique_vals = update_sparse_unique(train_data, data_info)
        cls.multi_sparse_unique_vals = update_multi_sparse_unique(train_data, data_info)
        if shuffle:
            train_data = cls.shuffle_data(train_data, seed)

        (
            merge_transformed,
            user_indices,
            item_indices,
            sparse_cols,
            multi_sparse_cols,
        ) = _build_transformed_set_feat(
            train_data,
            cls.user_unique_vals,
            cls.item_unique_vals,
            is_train=True,
            is_ordered=False,
            data_info=data_info,
        )
        sparse_offset = merge_offset(
            sparse_cols,
            multi_sparse_cols,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
        )
        sparse_oov = get_oov_pos(
            sparse_cols,
            multi_sparse_cols,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
        )

        all_sparse_col = data_info.sparse_col.name
        pad_val = (
            data_info.multi_sparse_combine_info.pad_val
            if cls.multi_sparse_unique_vals
            else dict()
        )
        multi_sparse_info = get_multi_sparse_info(
            all_sparse_col,
            cls.sparse_col,
            cls.multi_sparse_col,
            cls.sparse_unique_vals,
            cls.multi_sparse_unique_vals,
            pad_val,
        )

        _update_func = functools.partial(
            update_unique_feats,
            train_data,
            data_info,
            sparse_unique=cls.sparse_unique_vals,
            multi_sparse_unique=cls.multi_sparse_unique_vals,
            sparse_offset=sparse_offset,
            sparse_oov=sparse_oov,
        )
        user_sparse_unique, user_dense_unique = _update_func(
            unique_ids=cls.user_unique_vals, is_user=True
        )
        item_sparse_unique, item_dense_unique = _update_func(
            unique_ids=cls.item_unique_vals, is_user=False
        )

        interaction_data = train_data[["user", "item", "label"]]
        user_consumed, item_consumed = update_consumed(
            user_indices,
            item_indices,
            len(cls.user_unique_vals),
            len(cls.item_unique_vals),
            data_info,
            merge_behavior,
        )

        new_data_info = DataInfo(
            data_info.col_name_mapping,
            interaction_data,
            user_sparse_unique,
            user_dense_unique,
            item_sparse_unique,
            item_dense_unique,
            user_consumed,
            item_consumed,
            cls.user_unique_vals,
            cls.item_unique_vals,
            cls.sparse_unique_vals,
            sparse_offset,
            sparse_oov,
            cls.multi_sparse_unique_vals,
            multi_sparse_info,
            seed,
        )
        new_data_info.old_info = store_old_info(data_info)
        cls.train_called = True
        return merge_transformed, new_data_info


def _get_sparse_unique_vals(sparse_col, train_data):
    if not sparse_col:
        return
    sparse_unique_vals = dict()
    for col in sparse_col:
        sparse_unique_vals[col] = np.sort(train_data[col].unique())
    return sparse_unique_vals


def _get_multi_sparse_unique_vals(multi_sparse_col, train_data, pad_val):
    if not multi_sparse_col:
        return None, None
    multi_sparse_unique_vals = dict()
    if not isinstance(pad_val, (list, tuple)):
        pad_val = [pad_val] * len(multi_sparse_col)
    if len(multi_sparse_col) != len(pad_val):
        raise ValueError("Length of `multi_sparse_col` and `pad_val` doesn't match")
    pad_val_dict = dict()
    for i, field in enumerate(multi_sparse_col):
        unique_vals = set(itertools.chain.from_iterable(train_data[field].to_numpy().T))
        if pad_val[i] in unique_vals:
            unique_vals.remove(pad_val[i])
        # use name of a field's first column as representative
        multi_sparse_unique_vals[field[0]] = np.sort(list(unique_vals))
        pad_val_dict[field[0]] = pad_val[i]
    return multi_sparse_unique_vals, pad_val_dict


def _build_transformed_set(
    data,
    user_unique_vals,
    item_unique_vals,
    is_train,
    is_ordered,
    has_feats=False,
):
    user_indices, item_indices = get_id_indices(
        data,
        user_unique_vals,
        item_unique_vals,
        is_train,
        is_ordered,
    )
    if "label" in data.columns:
        labels = data["label"].to_numpy(dtype=np.float32)
    else:
        # in case test_data has no label column, create dummy labels for consistency
        labels = np.zeros(len(data), dtype=np.float32)

    if has_feats:
        return user_indices, item_indices, labels

    if is_train:
        transformed_data = TransformedSet(user_indices, item_indices, labels)
        return transformed_data, user_indices, item_indices
    else:
        return TransformedEvalSet(user_indices, item_indices, labels)


def _build_transformed_set_feat(
    data,
    user_unique_vals,
    item_unique_vals,
    is_train,
    is_ordered,
    data_info=None,
):
    user_indices, item_indices, labels = _build_transformed_set(
        data, user_unique_vals, item_unique_vals, is_train, is_ordered, has_feats=True
    )
    if not is_train:
        return TransformedEvalSet(user_indices, item_indices, labels)

    sparse_indices, dense_values, sparse_cols, multi_sparse_cols = _build_features(
        data, is_train, is_ordered, data_info
    )
    transformed_data = TransformedSet(
        user_indices, item_indices, labels, sparse_indices, dense_values
    )

    pure_data = transformed_data, user_indices, item_indices
    if not data_info:
        return pure_data + (sparse_indices, dense_values)  # noqa: RUF005
    else:
        return pure_data + (sparse_cols, multi_sparse_cols)  # noqa: RUF005


def _build_features(data, is_train, is_ordered, data_info):
    sparse_indices, dense_values = None, None
    if data_info:
        sparse_cols, multi_sparse_cols = recover_sparse_cols(data_info)
        dense_cols = data_info.dense_col.name
    else:
        sparse_cols = DatasetFeat.sparse_col
        multi_sparse_cols = DatasetFeat.multi_sparse_col
        dense_cols = DatasetFeat.dense_col

    sparse_unique = DatasetFeat.sparse_unique_vals
    multi_sparse_unique = DatasetFeat.multi_sparse_unique_vals
    if sparse_cols or multi_sparse_cols:
        sparse_indices = merge_sparse_indices(
            data,
            sparse_cols,
            multi_sparse_cols,
            sparse_unique,
            multi_sparse_unique,
            is_train,
            is_ordered,
        )
    if dense_cols:
        dense_values = data[dense_cols].to_numpy(dtype=np.float32)
    return sparse_indices, dense_values, sparse_cols, multi_sparse_cols


def _get_sparse_unique_vals_lazy(sparse_col, user_features, item_features):
    """Get sparse unique values from feature DataFrames for lazy loading."""
    if not sparse_col:
        return
    sparse_unique_vals = dict()
    for col in sparse_col:
        unique_vals = set()
        if user_features is not None and col in user_features.columns:
            unique_vals.update(user_features[col].unique())
        if item_features is not None and col in item_features.columns:
            unique_vals.update(item_features[col].unique())
        sparse_unique_vals[col] = np.sort(list(unique_vals))
    return sparse_unique_vals


def _get_multi_sparse_unique_vals_lazy(multi_sparse_col, user_features, item_features, pad_val):
    """Get multi-sparse unique values from feature DataFrames for lazy loading."""
    if not multi_sparse_col:
        return None, None
    multi_sparse_unique_vals = dict()
    if not isinstance(pad_val, (list, tuple)):
        pad_val = [pad_val] * len(multi_sparse_col)
    if len(multi_sparse_col) != len(pad_val):
        raise ValueError("Length of `multi_sparse_col` and `pad_val` doesn't match")
    pad_val_dict = dict()
    for i, field in enumerate(multi_sparse_col):
        unique_vals = set()
        for df in [user_features, item_features]:
            if df is None:
                continue
            for col in field:
                if col in df.columns:
                    unique_vals.update(df[col].unique())
        if pad_val[i] in unique_vals:
            unique_vals.remove(pad_val[i])
        # use name of a field's first column as representative
        multi_sparse_unique_vals[field[0]] = np.sort(list(unique_vals))
        pad_val_dict[field[0]] = pad_val[i]
    return multi_sparse_unique_vals, pad_val_dict


def _construct_unique_feat_lazy(
    user_unique_vals,
    item_unique_vals,
    user_features,
    item_features,
    sparse_unique_vals,
    multi_sparse_unique_vals,
    col_name_mapping,
    user_sparse_col_names,
    user_dense_col_names,
    item_sparse_col_names,
    item_dense_col_names,
):
    """Construct unique feature arrays from feature DataFrames for lazy loading.

    This function builds the unique feature matrices (user_sparse_unique, etc.)
    that are needed for looking up features during negative sampling.
    The sparse indices include offsets to match the non-lazy version.
    """
    user_sparse_unique = None
    user_dense_unique = None
    item_sparse_unique = None
    item_dense_unique = None

    # Build global column to offset mapping
    # Offset for each column is the cumulative sum of previous columns' vocab sizes + 1 (for oov)
    all_sparse_cols = list(col_name_mapping.get("sparse_col", {}).keys())
    all_multi_sparse_cols = []
    if "multi_sparse" in col_name_mapping:
        for field_cols in col_name_mapping["multi_sparse"].values():
            all_multi_sparse_cols.extend(field_cols)

    # Calculate offsets for each column
    col_offset = {}
    cumulative_offset = 0
    for col in all_sparse_cols:
        col_offset[col] = cumulative_offset
        if sparse_unique_vals and col in sparse_unique_vals:
            cumulative_offset += len(sparse_unique_vals[col]) + 1  # +1 for oov

    for col in all_multi_sparse_cols:
        col_offset[col] = cumulative_offset
        # Find the field for this column to get unique values
        field_name = None
        if "multi_sparse" in col_name_mapping:
            for fname, fcols in col_name_mapping["multi_sparse"].items():
                if col in fcols:
                    field_name = fname
                    break
        if field_name and multi_sparse_unique_vals and field_name in multi_sparse_unique_vals:
            # All columns in a multi_sparse field share the same vocab size
            cumulative_offset += len(multi_sparse_unique_vals[field_name]) + 1

    # Helper function to get sparse indices for a DataFrame
    def _get_sparse_indices(df, id_col, id_unique_vals, sparse_cols):
        if df is None or not sparse_cols:
            return None

        # Ensure df has proper index for lookup
        df_dedup = df.drop_duplicates(subset=[id_col], keep="last")
        df_dedup = df_dedup.set_index(id_col)

        # Build indices matrix
        n_ids = len(id_unique_vals)
        n_features = len(sparse_cols)
        indices = np.zeros((n_ids, n_features), dtype=np.int32)

        for j, col in enumerate(sparse_cols):
            if col not in df_dedup.columns:
                # Set to oov with offset for missing columns
                offset = col_offset.get(col, 0)
                if sparse_unique_vals and col in sparse_unique_vals:
                    oov_val = len(sparse_unique_vals[col])
                elif multi_sparse_unique_vals:
                    field_name = None
                    if "multi_sparse" in col_name_mapping:
                        for fname, fcols in col_name_mapping["multi_sparse"].items():
                            if col in fcols:
                                field_name = fname
                                break
                    if field_name and field_name in multi_sparse_unique_vals:
                        oov_val = len(multi_sparse_unique_vals[field_name])
                    else:
                        oov_val = 0
                else:
                    oov_val = 0
                indices[:, j] = oov_val + offset
                continue

            # Get offset for this column
            offset = col_offset.get(col, 0)

            # Determine which unique values dict to use
            if sparse_unique_vals and col in sparse_unique_vals:
                unique_vals = sparse_unique_vals[col]
            elif multi_sparse_unique_vals:
                # Find the field for this column
                field_name = None
                if "multi_sparse" in col_name_mapping:
                    for fname, fcols in col_name_mapping["multi_sparse"].items():
                        if col in fcols:
                            field_name = fname
                            break
                if field_name and field_name in multi_sparse_unique_vals:
                    unique_vals = multi_sparse_unique_vals[field_name]
                else:
                    continue
            else:
                continue

            oov_val = len(unique_vals)
            idx_mapping = dict(zip(unique_vals, range(len(unique_vals))))

            for i, uid in enumerate(id_unique_vals):
                if uid in df_dedup.index:
                    val = df_dedup.loc[uid, col]
                    indices[i, j] = idx_mapping.get(val, oov_val) + offset
                else:
                    indices[i, j] = oov_val + offset

        return indices

    # Helper function to get dense values
    def _get_dense_values(df, id_col, id_unique_vals, dense_cols):
        if df is None or not dense_cols:
            return None

        df_dedup = df.drop_duplicates(subset=[id_col], keep="last")
        df_dedup = df_dedup.set_index(id_col)

        n_ids = len(id_unique_vals)
        n_features = len(dense_cols)
        values = np.zeros((n_ids, n_features), dtype=np.float32)

        for j, col in enumerate(dense_cols):
            if col not in df_dedup.columns:
                continue
            for i, uid in enumerate(id_unique_vals):
                if uid in df_dedup.index:
                    values[i, j] = df_dedup.loc[uid, col]

        return values

    # Build user features
    if user_sparse_col_names:
        user_sparse_unique = _get_sparse_indices(
            user_features, "user", user_unique_vals, user_sparse_col_names
        )
    if user_dense_col_names:
        user_dense_unique = _get_dense_values(
            user_features, "user", user_unique_vals, user_dense_col_names
        )

    # Build item features
    if item_sparse_col_names:
        item_sparse_unique = _get_sparse_indices(
            item_features, "item", item_unique_vals, item_sparse_col_names
        )
    if item_dense_col_names:
        item_dense_unique = _get_dense_values(
            item_features, "item", item_unique_vals, item_dense_col_names
        )

    return user_sparse_unique, user_dense_unique, item_sparse_unique, item_dense_unique
