from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader

from clarity.enhancer.gha.gha_utils import format_gaintable, get_gaintable
from clarity.utils.audiogram import Listener
from clarity.utils.file_io import read_signal, write_signal


class GHAHearingAid:
    def __init__(
        self,
        sample_rate=44100,
        ahr=20,
        audf=None,
        cfg_file="prerelease_combination4_smooth",
        noise_gate_levels=None,
        noise_gate_slope=0,
        cr_level=0,
        max_output_level=100,
        equiv_0db_spl=100,
        test_nbits=16,
    ):
        if audf is None:
            audf = [250, 500, 1000, 2000, 3000, 4000, 6000, 8000]
        if noise_gate_levels is None:
            noise_gate_levels = [38, 38, 36, 37, 32, 26, 23, 22, 8]

        self.sample_rate = sample_rate
        self.ahr = ahr
        self.audf = audf
        self.cfg_file = cfg_file
        self.noise_gate_levels = noise_gate_levels
        self.noise_gate_slope = noise_gate_slope
        self.cr_level = cr_level
        self.max_output_level = max_output_level
        self.equiv_0db_spl = equiv_0db_spl
        self.test_nbits = test_nbits

    def create_configured_cfgfile(
        self, input_file, output_file, formatted_sGt, cfg_template_file
    ):
        """Using Jinja2, generates cfg file for given configuration.

        Creates template output file and configures with correct filenames, peak level
        out and DC gaintable.

        Args:
            input_file (str): file to process
            output_file (str): file in which to store processed file
            formatted_sGt (ndarray): gaintable formatted for input into cfg file
            cfg_template_file: configuration file template
            ahr (int): amplification headroom

        Returns:
            cfg_filename (str): cfg filename
        """

        if self.sample_rate != 44100:
            logging.error("Current GHA configuration requires 44.1kHz sampling rate.")
            raise ValueError(
                "Current GHA configuration requires 44.1kHz sampling rate."
            )

        cfg_template_file = pathlib.Path(cfg_template_file)

        # Define cfg filenames
        # Read new file and replace any parameter values necessary
        # Update peaklevel out by adding headroom
        logging.info("Adding %s dB headroom", self.ahr)

        peaklevel_in = int(self.equiv_0db_spl)
        peaklevel_out = int(self.equiv_0db_spl + self.ahr)

        # Render jinja2 template
        file_loader = FileSystemLoader(cfg_template_file.parent)
        env = Environment(loader=file_loader)
        template = env.get_template(cfg_template_file.name)
        output = template.render(
            io_in=input_file,
            io_out=output_file,
            peaklevel_in=(
                f"[{peaklevel_in} {peaklevel_in} {peaklevel_in} {peaklevel_in}]"
            ),
            peaklevel_out=f"[{peaklevel_out} {peaklevel_out}]",
            gtdata=formatted_sGt,
        )

        return output

    def process_files(
        self, infile_names: list[str], outfile_name: str, listener: Listener
    ):
        """Process a set of input signals and generate an output.

        Args:
            infile_names (list[str]): List of input wav files. One stereo wav
                file for each hearing device channel
            outfile_name (str): File in which to store output wav files
            dry_run (bool): perform dry run only
        """
        logging.info("Processing %s with listener %s", outfile_name, listener.id)

        logging.info(
            "Audiogram severity is %s (left) and %s (right)",
            listener.audiogram_left.severity,
            listener.audiogram_right.severity,
        )
        audiogram_left = listener.audiogram_left.resample(self.audf)
        audiogram_right = listener.audiogram_right.resample(self.audf)

        # Get gain table with noisegate correction
        gaintable = get_gaintable(
            audiogram_left,
            audiogram_right,
            self.noise_gate_levels,
            self.noise_gate_slope,
            self.cr_level,
            self.max_output_level,
        )
        formatted_sGt = format_gaintable(gaintable, noisegate_corr=True)

        cfg_template = Path(__file__).parent / f"cfg_files/{self.cfg_file}_template.cfg"

        # Merge CH1 and CH3 files. This is the baseline configuration.
        # CH2 is ignored.
        fd_merged, merged_filename = tempfile.mkstemp(
            prefix="clarity-merged-", suffix=".wav"
        )
        # Only need file name; must immediately close the unused file handle.
        os.close(fd_merged)

        self.create_HA_inputs(infile_names, merged_filename)

        # Create the openMHA config file from the template
        fd_cfg, cfg_filename = tempfile.mkstemp(
            prefix="clarity-openmha-", suffix=".cfg"
        )
        # Again, only need file name; must immediately close the unused file handle.
        os.close(fd_cfg)
        with open(cfg_filename, "w", encoding="utf-8") as f:
            f.write(
                self.create_configured_cfgfile(
                    merged_filename, outfile_name, formatted_sGt, cfg_template
                )
            )

        # Process file using configured cfg file
        # Suppressing OpenMHA output with -q - comment out when testing
        # Append log of OpenMHA commands to /cfg_files/logfile
        subprocess.run(
            [
                "mha",
                "-q",
                "--log=logfile.txt",
                f"?read:{cfg_filename}",
                "cmd=start",
                "cmd=stop",
                "cmd=quit",
            ],
            check=True,
        )

        # Delete temporary files.
        os.remove(merged_filename)
        os.remove(cfg_filename)

        # Check output signal has energy in every channel
        sig = read_signal(
            outfile_name, sample_rate=self.sample_rate, allow_resample=False
        )

        if len(np.shape(sig)) == 1:
            sig = np.expand_dims(sig, axis=1)

        if not np.all(np.sum(abs(sig), axis=0)):
            raise ValueError("Channel empty.")

        # Rewriting as floating point
        write_signal(outfile_name, sig, self.sample_rate, floating_point=True)

        logging.info("OpenMHA processing complete")

    def create_HA_inputs(self, infile_names: list[str], merged_filename: str) -> None:
        """Create input signal for baseline hearing aids.

        The baseline hearing aid takes a 4-channel wav file as input. This is
        constructed from the left and right signals of the front (CH1) and
        rear (CH3) microphones that are available in the Clarity data.

          Args:
              infile_names (list[str]): Names of file to read
              merged_file_name (str): Name of file to write

          Raises:
              ValueError: If input channel names are inconsistent
        """
        if (infile_names[0][-5] != "1") or (infile_names[2][-5] != "3"):
            raise ValueError("HA-input signal error: channel mismatch!")

        signal_CH1 = read_signal(
            infile_names[0], sample_rate=self.sample_rate, allow_resample=False
        )
        signal_CH3 = read_signal(
            infile_names[2], sample_rate=self.sample_rate, allow_resample=False
        )

        merged_signal = np.zeros((len(signal_CH1), 4))

        # channel index 0 = front microphone on the left hearing aid
        merged_signal[:, 0] = signal_CH1[:, 0]
        # channel index 1 = front microphone on the right hearing aid
        merged_signal[:, 1] = signal_CH1[:, 1]
        # channel index 2 = rear microphone on the left hearing aid
        merged_signal[:, 2] = signal_CH3[:, 0]
        # channel index 3 = rear microphone on the right hearing aid
        merged_signal[:, 3] = signal_CH3[:, 1]

        write_signal(
            merged_filename,
            merged_signal,
            self.sample_rate,
            floating_point=True,
            strict=True,
        )
