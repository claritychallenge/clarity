## Machine learning challenges for hearing aid processing.
<p align="center">
  <img src="docs/images/earfinal_clarity_customColour.png" alt="drawing" width="250"/>
</p>

We are organising a series of machine learning challenges to enhance hearing-aid signal processing and to better predict how people perceive speech-in-noise. For further details of the Clarity Project visit [the Clarity project website](http://claritychallenge.org/), and for details of our latest challenge visit our [challenge documentation site](https://claritychallenge.github.io/clarity_CC_doc/). You can contact the Clarity Team by email at [claritychallengecontact@gmail.com](claritychallengecontact@gmail.com).

In this repository, you will find code to support all Clarity Challenges, including baselines, toolkits, and systems from participants. **We encourage you to make your system/model open source and contribute to this repository.**

### The 2nd Clarity Enhancement Challenge (CEC2) has launched! [Take part](https://claritychallenge.github.io/clarity_CC_doc/docs/category/taking-part):fire::fire::fire:

## Installation

```bash
# First clone the repo
git clone https://github.com/claritychallenge/clarity.git
cd clarity

# Second create & activate environment with conda, see https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
conda create --name clarity python=3.8
conda activate clarity

# Last install with pip
pip install -e .
```

## Challenges

Current challenge

- [The 2nd Clarity Enhancement Challenge (CEC2)](./recipes/cec2)

Previous challenges

- [The 1st Clarity Prediction Challenge (CPC1)](./recipes/cpc1)
- [The 1st Clarity Enhancement Challenge (CEC1)](./recipes/cec1/baseline)


## Open-source systems
- CEC1:
  - [E009 Sheffield Entry](./recipes/cec1/e009_sheffield)
