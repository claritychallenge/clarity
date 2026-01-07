# MSBG Hearing Loss Simulator

This distribution is an implementation of the MSBG (Moore, Stone, Baer, and Glasberg ) hearing-loss simulator from Cambridge University. For more details see this
[blog post](https://claritychallenge.org/blog/Hearing%20loss%20simulation).

## History

The first round of the Clarify Challenge needed an objective measure of hearing-impaired intelligibility. The aim was not to define a definitive perceptual model, but simply to have a principled way of ranking enhancement systems. The idea was to combine a hearing-loss simulation with STOI.

Michael Stone and Brian Moore’s hearing-loss simulation had a MATLAB/C++ version that had been shared with several groups outside Cambridge, and Michael also had a more recent, fully MATLAB implementation. With Brian’s consent, Michael was happy to share that version for the purposes of the challenge.

Trevor Cox and Simone Graetzer translated the MATLAB code into Python. The translation is a very literal, function-by-function port, and it has been carefully validated against the MATLAB implementation through regression testing, with identical design choices retained throughout.

The original MATLAB code itself was unpublished and does not have a single, definitive reference. From the comments and structure, it appears to have evolved over a long period (roughly 2005–2015, possibly later), with different components validated in different papers. We preserved all original comments in the Python version, including references to papers where specific parts of the model are mentioned.

## References

Spectral smearing:

T. Baer and B.C.J. Moore, Effects of spectral smearing on the intelligibility of sentences in the presence of noise, J. Acoust. Soc. Am. 94: 1229-1241 (1993). [PubMed](https://pubmed.ncbi.nlm.nih.gov/8201124/)

T. Baer and B.C.J. Moore, Effects of spectral smearing on the intelligibility of sentences in the presence of interfering speech, J. Acoust. Soc. Am. 95: 2277-2280 (1994).   [PubMed](https://pubmed.ncbi.nlm.nih.gov/8201124/)

Recruitment simulation:

B.C.J. Moore and B.R. Glasberg, Simulation of the effects of loudness recruitment and threshold elevation on the intelligibility of speech in quiet and in a background of speech, J. Acoust. Soc. Am. 94: 2050-2062 (1993). [PubMed](https://pubmed.ncbi.nlm.nih.gov/8227747/)

Filtering to equalise signal for cochlear presentation:

M.A. Stone and B.C.J. Moore, Tolerable hearing-aid delays. I. Estimation of limits imposed by the auditory path alone using simulated hearing losses, Ear Hear. 20: 182-192 (1999). [PubMed](https://pubmed.ncbi.nlm.nih.gov/10386846/)

## Matlab Comments

Here are selected parts of the header for original Matlab main control program.

Hearing impairment simulator, written by members of Auditory Perception Group, University of Cambridge, ca 1991-2013.

In this version, user specifies a file in variable "IpFileNames", and this is processed for as many channels as there are in
each signal file for as many times as there are audiograms in "Audiogram_Master()" array.
simulation is only intended for audiogram max of 80-90 dB HL, otherwise things go wrong/unrealistic')(excessive recruitment)

Since 2007 this code maintained and updated by Michael A Stone, <Michael.Stone@manchester.ac.uk>
Comments extended for AMToolbox version, Jan 2025
further updated by M.A.Stone, Oct 2015 (more extensive comments)
2013: converted to replace DOS C executables with MATLAB script. Filterbank made higher resolution than previous
and filter now overlap more than previous (previously overlapped at -6dB, now ~33%  hypersampled.
Extensive header updated Jan 2011.
modified Oct 09 to do multi-channel (stereo) processing
modified by MAS, Jun09 for 44.1k sampling,  new executables as well: same old names
modified by MAS, Oct08 to try to get more consistent results, for accessibility project, Feb 2007.

modified by MAStone, Nov 2013 to remove previous dependency on C executables to perform filterbank and recruitment.
this was originaly done because of computer euqipment available in the early 1990s.
Now all included as MATLAB scripts.  Should lead to less problems with head- (and floor-) room and more clarity in calibration
Also modified the whole software package to move reference-to-eardrum filtering (eg ff/df/ITU) into this script.

PROCESSING SUMMARY JAN 2025
(1) take wav file from "IpFileNames" variable, measure RMS and adjust to reference calibration level as well as incorporating
any required change in input level from 65 dB SPL reference eg uif sijmulating at a different input level, prepend calibration
tone and calibration speech-spectrum-shaped noise at start of each channel signal for each audiogram in array "Audiogram_Master",
& process file........
(2) send to MATLAB function that applies linear filtering to simulate passage from ITU/diffuse/free-field to cochlea
         (mainly a bass cut and a mid-range boost, does not usually increase signal level)
(3) [obsolete 2013] write cochlea-referenced signal as 16 bit binary file for processing by DOS executable called by batch file
(4) impairment simulation is done in two stages: spectral smearing (optional switch to include this now provided in function call)
    and  recruitment simulation (obligatory)
(5) re-call MATLAB function of (2) above so as to apply INVERSE linear filtering from that in (2)
         (to simulate passage from cochlea to ITU/diffuse/free-field, does increase signal level)
(6) write processed signal out to wav file: filename is automatically generated by adding the index of the audiogram to the input filename [AudiogX]
(7) loop back to process next audiogram for this file (until audiograms exhausted), and/or select next file (until filelist exhausted)

In the original recruitment simulation (1990s), the filter bandwidth was set to 3 times broader than "normal" (ie 3*ERBn).  This sort of width is suitable for severe losses, where signal bandwidth is usually limited, hence the original software was written for 16 kHz samplerate, and used 13 overlapping filters, at 3-ERBN spacing. This version, with samplerate set to 44.1 kHz was intended for more moderate losses, hence the recruitment filterbank adjusts, depending on the mean hearing loss to be simulated, caclculated in the range 2-8 kHz.

Three degrees of broadening are available, x1.5, x2 and x3 broadening (see "cochlear_simulate.m"), but now with greater overlap of filters (~33% hyper-sampled, 1.1-, 1.5- & 2.3-ERBn spaced).

The degree of spectral smearing, applied uniformly across frequencies, also depends on the average audiogram between 2 and 8 kHz.  The higher the degree of smearing, the more "fuzziness” it applies to the signal, by bringing up the level of noise between signal components.  To a first order, this can be regarded as a form of degradation of IHC signal quality. Because of the non-linearity of the processing, and the overlap of the filters, this software does not (and cannot) provide an "exact" simulation, more a qualitative feel for the sorts of problems to be experienced.  So if a calibration tone comes out 1-2 dB away from where one expected it to, then do not be surprised.
