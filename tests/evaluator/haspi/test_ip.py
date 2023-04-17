"""Tests for ip module"""
import numpy as np
import pytest

from clarity.evaluator.haspi.ip import (
    get_neural_net,
    nn_feed_forward,
    nn_feed_forward_ensemble,
)


def test_get_neural_net() -> None:
    """Test for get_neural_net function"""

    # Retrieve the standard HASPI neural network
    (
        neural_net_params,
        weights_hidden,
        weights_out,
        normalization_factor,
    ) = get_neural_net()

    # Check that the correct parameters exist
    assert neural_net_params.keys() == {
        "input_layer",
        "hidden_layer",
        "output_layer",
        "activation_function",
        "offset_activation",
        "maximum_activation",
    }

    n_hidden_nodes = neural_net_params["hidden_layer"]
    n_output_nodes = neural_net_params["output_layer"]
    n_input_nodes = neural_net_params["input_layer"]

    # Check weights are correct shape for the number of nodes
    assert len(weights_out[0]) == n_hidden_nodes + 1
    assert len(weights_out[0][0]) == n_output_nodes
    assert len(weights_hidden[0]) == n_input_nodes + 1
    assert len(weights_hidden[0][0]) == n_hidden_nodes
    # Check that the number of networks in the ensembles match
    assert len(weights_out) == len(weights_hidden)
    # Check the specific values of the weights are correct
    assert normalization_factor == 0.9508
    assert np.sum(weights_out) == pytest.approx(
        -33.8515, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.sum(weights_hidden) == pytest.approx(
        -670.82009999, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_nn_feed_forward_ensemble() -> None:
    """Test for nn_feed_forward_ensemble function"""

    # Retrieve the standard HASPI neural network
    (
        neural_net_params,
        weights_hidden,
        weights_out,
        _normalization_factor,
    ) = get_neural_net()

    # Make some fake data matching the input layer size
    data = np.ones((1, neural_net_params["input_layer"]))

    # Check that the output is correct
    x = nn_feed_forward_ensemble(data, neural_net_params, weights_hidden, weights_out)
    assert x == pytest.approx(
        0.95078679, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )


def test_nn_feed_forward() -> None:
    """Test for nn_feed_forward function"""

    # Retrieve the standard HASPI neural network
    (
        neural_net_params,
        weights_hidden,
        weights_out,
        _normalization_factor,
    ) = get_neural_net()

    # Make some fake data matching the input layer size
    data = np.ones(neural_net_params["input_layer"])

    # Check that the output is correct for first network in ensemble
    hidden, output = nn_feed_forward(
        data, neural_net_params, weights_hidden[0], weights_out[0]
    )
    # Note, it adds a constant 1 to the start of the output and hidden layers
    assert hidden.shape == (neural_net_params["hidden_layer"] + 1,)
    assert output.shape == (neural_net_params["output_layer"] + 1,)
    assert hidden[0] == pytest.approx(
        1.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert output[0] == pytest.approx(
        1.0, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    # Check that the mean of the hidden and output layers are correct
    assert np.mean(hidden) == pytest.approx(
        0.599999906437356, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
    assert np.mean(output) == pytest.approx(
        0.982229820658455, rel=pytest.rel_tolerance, abs=pytest.abs_tolerance
    )
