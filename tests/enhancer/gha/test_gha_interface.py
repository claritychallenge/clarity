"""Tests for enhancer.gha.gha_interface module"""
import hashlib

import numpy as np
import pytest

from clarity.enhancer.gha.gha_interface import GHAHearingAid
from clarity.utils.audiogram import Audiogram, Listener
from clarity.utils.file_io import read_signal, write_signal


def setup_ha_and_data(root_path):
    """Setup the hearing aid and make some dummy data file for testing

    Args:
        root_path (pathlib.Path): Path to temporary directory

    Returns:
        tuple: (gha_hearing_aid, stereo_signal, tmp_filenames)
            gha_hearing_aid (GHAHearingAid): Hearing aid object
            stereo_signal (np.ndarray): Dummy stereo signal stored in the files
            tmp_filenames (list): List of temporary filenames
    """
    gha_hearing_aid = GHAHearingAid()

    tmp_filenames = [str(root_path / f"xxx.CH{chan}.wav") for chan in [1, 2, 3, 0]]

    stereo_signal = np.array(
        [[-0.1, 0.1, -0.1, 0.1, 0.0, -1.0], [-0.2, 0.2, -0.2, 0.2, 0.0, 1.1]]
    ).T  # <- signals are store with channels as columns

    for filename in tmp_filenames:
        write_signal(
            filename,
            stereo_signal,  # <- signals are stored with channels as columns
            gha_hearing_aid.sample_rate,
            floating_point=True,
        )
    return gha_hearing_aid, stereo_signal, tmp_filenames


def test_gha_hearing_aid_init():
    """test the the gha hearing aid is initialized correctly"""
    gha_hearing_aid = GHAHearingAid()
    assert len(gha_hearing_aid.audf) == 8
    assert len(gha_hearing_aid.noise_gate_levels) == 9


def test_create_configured_cfgfile():
    """test that the cfgfile can be correctly constructed"""
    gha_hearing_aid = GHAHearingAid()
    input_file = "input.wav"
    output_file = "output.wav"
    formatted_sGt = np.array([1, 2, 3])  # Some dummy data which will appear in output
    cfg_template_file = (
        "clarity/enhancer/gha/cfg_files/prerelease_combination3_smooth_template.cfg"
    )
    output = gha_hearing_aid.create_configured_cfgfile(
        input_file, output_file, formatted_sGt, cfg_template_file
    )

    # Check that the output is as expected
    assert (
        hashlib.md5(output.encode("utf-8")).hexdigest()
        == "b49533003cd6449fa8c4c5edacaef067"
    )


def test_create_configured_cfgfile_error():
    """Check gives error if sampling rate is not 44.1kHz"""
    gha_hearing_aid = GHAHearingAid()
    gha_hearing_aid.sample_rate = 16000  # <-- any value other than 44100
    with pytest.raises(ValueError):
        arbitrary_data = np.array([1, 2, 3])
        gha_hearing_aid.create_configured_cfgfile(
            "input.wav", "output.wav", arbitrary_data, "template.cfg"
        )


def test_process_files(mocker, tmp_path):
    """test that the gha process can run"""

    gha_hearing_aid, _stereo_signal, infile_names = setup_ha_and_data(tmp_path)

    levels = np.array([45, 45, 35, 45, 60, 65])
    frequencies = np.array([250, 500, 1000, 2000, 4000, 6000])
    audiogram = Audiogram(levels=levels, frequencies=frequencies)
    listener = Listener(audiogram_left=audiogram, audiogram_right=audiogram)
    # Mock the subprocess.run function as OpenMHA is not installed
    m = mocker.patch("clarity.enhancer.gha.gha_interface.subprocess.run")

    # write a dummy output file so that process_files still runs
    # despite that the fact that the file generation is mocked out
    outfile_name = tmp_path / "output.wav"

    write_signal(
        outfile_name,
        np.array([[-0.1, 0.1], [-0.2, 0.2]]),
        sample_rate=gha_hearing_aid.sample_rate,
        floating_point=True,
    )

    gha_hearing_aid.process_files(
        infile_names=infile_names,
        outfile_name=outfile_name,
        listener=listener,
    )
    # Check that the subprocess.run function was called
    m.assert_called_once()


def test_create_ha_inputs(tmp_path):
    """test that the hearing aid inputs can be created correctly"""

    gha_hearing_aid, stereo_signal, tmp_filenames = setup_ha_and_data(tmp_path)
    # Do the merge with the test data files
    output_filename = "output.wav"  # tmp_path / "output.wav"
    gha_hearing_aid.create_HA_inputs(tmp_filenames, str(output_filename))

    # Read the output file and check it is correct
    result = read_signal(output_filename, sample_rate=gha_hearing_aid.sample_rate)
    assert result.shape == (stereo_signal.shape[0], 4)
    assert np.sum(result) == pytest.approx(np.sum(stereo_signal) * 2.0)


@pytest.mark.parametrize(
    "bad_names, error_type",
    [
        (["xxx.CH1.wav", "xxx.CH3.wav"], IndexError),
        (["xxx.CH1.wav", "x", "xxx.CH1.wav"], ValueError),
        (["xxx.CH1.wav", "x", "xxx.CH3.wa"], ValueError),
        (["xxx.CH4.wav", "x", "xxx.CH3.wav"], ValueError),
    ],
)
def test_create_ha_inputs_error(bad_names, error_type):
    """test that raises errror if file names don't follow the rules"""

    # Must be at least three names and first name must end "1????" and
    # third name must end "3????". It's a pretty weak test but it
    # at least prevents channels being supplied in the wrong order
    # would make more sense if the function constructed the names itself!
    gha_hearing_aid = GHAHearingAid()
    with pytest.raises(error_type):
        gha_hearing_aid.create_HA_inputs(bad_names, "output.wav")
