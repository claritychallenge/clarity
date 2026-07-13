# E009 - Implementation of the Sheffield entry for the first Clarity enhancement challenge (CEC1)

This repository contains the PyTorch implementation of "[A Two-Stage End-to-End System for Speech-in-Noise Hearing Aid Processing](https://claritychallenge.github.io/clarity2021-workshop/papers/Clarity_2021_paper_tu.pdf)", the Sheffield entry E009 for the first [Clarity enhancement challenge (CEC1)](https://claritychallenge.github.io/clarity2021-workshop/results.html). The system consists of a [Conv-TasNet](https://github.com/kaituoxu/Conv-TasNet) based denoising module, and a finite-inpulse-response (FIR) filter based amplification module. A differentiable approximation to the [Cambridge MSBG model](https://github.com/claritychallenge/clarity_CEC1/tree/master/projects/MSBG) released in the CEC1 is used in the loss function.

## Train

To build the overall system, the multi-channel Conv-TasNet based denoising module is trained in the first stage, and the FIR based amplification module is trained in the second stage. The FIR amplification module is dependent on the listener ID. To run the script, specify `path.cec1_root` as the CEC1 data path, and `path.exp_folder` as the experiment directory.

## References

* [1] Luo Y, Mesgarani N. Conv-tasnet: Surpassing ideal time–frequency magnitude masking for speech separation[J]. IEEE/ACM transactions on audio, speech, and language processing, 2019, 27(8): 1256-1266.
* [2] Andersen A H, de Haan J M, Tan Z H, et al. Refinement and validation of the binaural short time objective intelligibility measure for spatially diverse conditions[J]. Speech Communication, 2018, 102: 1-13.
* [3] Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. An algorithm for intelligibility prediction of time–frequency weighted noisy speech. IEEE Transactions on Audio, Speech, and Language Processing, 19(7), 2125-2136.
* [4] Zhang, J., Zorilă, C., Doddipatla, R., & Barker, J. On end-to-end multi-channel time domain speech separation in reverberant environments. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 6389-6393). IEEE.

## Citation

If you use this work, please cite:

```bibtex
@article{tutwo,
  title={A Two-Stage End-to-End System for Speech-in-Noise Hearing Aid Processing},
  author={Tu, Zehai and Zhang, Jisi and Ma, Ning and Barker, Jon},
  year={2021},
  booktitle={The Clarity Workshop on Machine Learning Challenges for Hearing Aids (Clarity-2021)},
}
```
