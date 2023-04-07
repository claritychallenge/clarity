# CLARITY CEC1 Baseline

In `data_config.yaml`, specify `path.root` folder in which you unpack the data; speciy `path.exp_folder` which will store intermediate signals and final results. Please be carefull if you want to change the parameters in the config file.

## 1. Hearing aid baseline system

THe openMHA software is used as the baseline hearing aid system. You need to [download](http://www.openmha.org/download/) and install openMHA to run `enhance.py`, which generates enhanced scenes.

## 2. Evaluation

The combination of MSBG hearing loss model and MBSTOI.  In `evaluate.py`, `run_HL_processing` runs MSBG hearing loss simulation, and `run_calculate_SI` computes MBSTOI scores. The intermediate signals and final scores will be saved. Note: if there are not `enhanced_signals`, the evaluation will be done with the original scene `mixed_CH0`.

## 3.Information on default parameter values

The default sampling rate is 44.1 kHz. The default output gain constant is 1. The default pre-duration and post-duration interferer periods are 2.0 and 1.0 (s), repectively. This means that the interferer starts 2 seconds before the target speech, and ends 1 second afterwards. The reverberation tail duration is 0.2 (s). The default duration of the ramp into and out of the interferer signal sampled for each scene is 0.5 (s).

The GHA and MSBG modules have parameter settings as follows: the default centre frequencies for the audiograms are [250, 500, 1000, 2000, 3000, 4000, 6000, 8000], as these are used in the listener data provided.

In our implementation, the CAMFIT prescription used in the GHA module sets parameter noisegatelevels (compression thresholds) to the band levels for a speech-shaped noise signal with an overall level of 45 dB SPL: [38, 38, 36, 37, 32, 26, 23, 22, 8]. There is a one-to-one input-output gain relationship below the compression threshold, hence for GHA parameter noisegateslope is set to 0. The input level for compression ratio calculation, parameter cr_level, is set to 0, indicating that it varies depending on insertion gains. The prescription is set to have a maximum output level, parameter max_output_level, of 100 dB SPL. The default configuration file for GHA is set to 'prerelease_combination3_smooth'.

In the baseline, a convention is used to estimate the level of signals as they pass through the pipeline: that a +/-1 square wave has RMS = 0dB (full scale or FS). Further, up to the point of signals being produced by the GHA module, the convention is that 0 dB FS is equal to 100 dB SPL. See parameter equiv0dBSPL. From that point onwards, as GHA provides 20 dB of amplification headroom (parameter ahr), the convention is that 0 dB FS is equal to 120 dB SPL. The MSBG signal processing relies on knowing the true input level in dB SPL.

In the creation of the provided scene metadata, signal-to-noise ratios (SNRs) were pseudo- randomly sampled from the ranges [0,12] dB for speech interferers, and [-6,6] dB for non- speech interferers. Three listeners are assigned to each scene.
