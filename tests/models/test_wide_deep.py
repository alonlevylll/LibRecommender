import sys

import pytest
import tensorflow as tf

from libreco.algorithms import WideDeep
from tests.models.utils_tf import ptest_tf_variables
from tests.utils_data import set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_multi_sparse_models import fit_multi_sparse
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize(
    "task, loss_type, sampler, neg_sampling",
    [
        ("rating", "focal", "random", None),
        ("rating", "focal", None, True),
        ("rating", "focal", "random", True),
        ("ranking", "cross_entropy", "random", False),
        ("ranking", "focal", "unconsumed", False),
        ("ranking", "cross_entropy", "random", True),
        ("ranking", "cross_entropy", "unconsumed", True),
        ("ranking", "focal", "popular", True),
        ("ranking", "unknown", "popular", True),
    ],
)
@pytest.mark.parametrize(
    "lr, lr_decay, reg, num_neg, use_bn, dropout_rate, hidden_units, num_workers",
    [
        ({"wide": 0.01, "deep": 3e-4}, False, None, 1, False, None, 1, 0),
        (None, True, 0.001, 3, True, 0.5, (32, 16), 2),
        (0.01, True, 0.001, 3, True, 0.5, 1, 0),
    ],
)
def test_wide_deep(
    feat_data_small,
    task,
    loss_type,
    sampler,
    neg_sampling,
    lr,
    lr_decay,
    reg,
    num_neg,
    use_bn,
    dropout_rate,
    hidden_units,
    num_workers,
):
    if not sys.platform.startswith("linux") and num_workers > 0:
        pytest.skip(
            "Windows and macOS use `spawn` in multiprocessing, which does not work well in pytest"
        )
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = feat_data_small
    if task == "ranking" and neg_sampling is False and loss_type == "cross_entropy":
        set_ranking_labels(train_data)
        set_ranking_labels(eval_data)

    if lr == 0.01:
        with pytest.raises(AssertionError):
            _ = WideDeep(task, data_info, loss_type, lr=lr)
    elif neg_sampling is None:
        with pytest.raises(AssertionError):
            WideDeep(task, data_info).fit(train_data, neg_sampling)
    elif task == "rating" and neg_sampling:
        with pytest.raises(ValueError):
            WideDeep(task, data_info).fit(train_data, neg_sampling)
    elif loss_type == "focal" and (neg_sampling is False or sampler is None):
        with pytest.raises(ValueError):
            WideDeep(task, data_info, sampler=sampler).fit(train_data, neg_sampling)
    elif task == "ranking" and loss_type not in ("cross_entropy", "focal"):
        with pytest.raises(ValueError):
            WideDeep(task, data_info, loss_type).fit(train_data, neg_sampling)
    else:
        model = WideDeep(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            embed_size=4,
            n_epochs=1,
            lr=lr,
            lr_decay=lr_decay,
            reg=reg,
            batch_size=80,
            sampler=sampler,
            num_neg=num_neg,
            use_bn=use_bn,
            dropout_rate=dropout_rate,
            hidden_units=hidden_units,
            tf_sess_config=None,
        )
        model.fit(
            train_data,
            neg_sampling,
            verbose=2,
            shuffle=True,
            eval_data=eval_data,
            metrics=get_metrics(task),
            eval_user_num=200,
            num_workers=num_workers,
        )
        ptest_tf_variables(model)
        ptest_preds(model, task, pd_data, with_feats=True)
        ptest_recommends(model, data_info, pd_data, with_feats=True)
        with pytest.raises(ValueError):
            model.recommend_user(1, 7, seq=[1, 2, 3])


def test_wide_deep_multi_sparse(prepare_multi_sparse_data):
    task = "ranking"
    pd_data, train_data, eval_data, data_info = prepare_multi_sparse_data
    model = fit_multi_sparse(
        WideDeep, train_data, eval_data, data_info, lr={"wide": 0.01, "deep": 3e-4}
    )
    ptest_preds(model, task, pd_data, with_feats=True)
    ptest_recommends(model, data_info, pd_data, with_feats=True)

    # test save and load model
    loaded_model, loaded_data_info = save_load_model(WideDeep, model, data_info)
    ptest_preds(loaded_model, task, pd_data, with_feats=True)
    ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=True)


def test_wide_deep_softmax_loss(feat_data_small):
    """Test WideDeep with softmax loss for rating task.
    
    Rating labels are automatically extracted from training data.
    """
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = feat_data_small
    
    model = WideDeep(
        task="rating",
        data_info=data_info,
        loss_type="softmax",
        embed_size=4,
        n_epochs=2,
        lr={"wide": 0.01, "deep": 3e-4},
        batch_size=80,
        hidden_units=(16, 8),
    )
    model.fit(
        train_data,
        neg_sampling=False,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=get_metrics("rating"),
    )
    
    # Rating labels should be auto-detected from training data
    assert model.rating_labels is not None
    assert model.n_rating_classes > 0
    n_classes = model.n_rating_classes
    
    # Test predictions (should use weighted average)
    ptest_preds(model, "rating", pd_data, with_feats=True)
    ptest_recommends(model, data_info, pd_data, with_feats=True)
    
    # Test predict_proba method
    user_id = pd_data["user"].iloc[0]
    item_id = pd_data["item"].iloc[0]
    proba_result = model.predict_proba(user_id, item_id)
    
    assert "labels" in proba_result
    assert "probabilities" in proba_result
    assert len(proba_result["labels"]) == n_classes
    assert proba_result["probabilities"].shape == (1, n_classes)
    # Probabilities should sum to 1
    assert abs(proba_result["probabilities"].sum() - 1.0) < 1e-5
    
    # Test batch predict_proba
    users = pd_data["user"].iloc[:5].tolist()
    items = pd_data["item"].iloc[:5].tolist()
    batch_proba = model.predict_proba(users, items)
    assert batch_proba["probabilities"].shape == (5, n_classes)


def test_wide_deep_softmax_validation():
    """Test validation for softmax loss configuration."""
    tf.compat.v1.reset_default_graph()
    
    # Test that softmax requires rating task
    with pytest.raises(ValueError, match="Softmax loss is only supported for rating task"):
        WideDeep(
            task="ranking",
            data_info=None,
            loss_type="softmax",
        )