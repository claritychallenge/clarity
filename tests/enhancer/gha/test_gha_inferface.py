"""Tests for enhancer.gha.gha_interface module"""
import numpy as np
import pytest

from clarity.enhancer.gha.gha_interface import GHAHearingAid


def test_gha_hearing_aid_init():
    """test the the gha hearing aid is initialized correctly"""
    gha_hearing_aid = GHAHearingAid()
    assert len(gha_hearing_aid.audf) == 8
    assert len(gha_hearing_aid.noisegatelevels) == 9


@pytest.mark.skip(reason="not fully implemented")
def test_create_configured_cfgfile():
    """test that the cfgfile can be correctly constructed"""
    gha_hearing_aid = GHAHearingAid()
    input_file = "input.wav"
    output_file = "output.wav"
    formatted_sGt = np.array([1, 2, 3])
    cfg_template_file = "template.cfg"
    gha_hearing_aid.create_configured_cfgfile(
        input_file, output_file, formatted_sGt, cfg_template_file
    )


@pytest.mark.skip(reason="not implemented")
def test_process_files():
    """test that the gha process can run"""


@pytest.mark.skip(reason="not implemented")
def test_read_and_write_signals():
    """test that signals can be read and written correctly"""


@pytest.mark.skip(reason="not implemented")
def test_create_ha_inputs():
    """test that the hearing aid inputs can be created correctly"""
