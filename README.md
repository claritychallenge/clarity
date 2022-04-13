## Machine learning challenges for hearing aid processing.
<p align="center">
  <img src="docs/images/earfinal_clarity_customColour.png" alt="drawing" width="250"/>
</p>

We are organising a series of machine learning challenges to enhance hearing-aid signal processing and to better predict how people perceive speech-in-noise. For further details of the Clarity Project visit [the Clarity Challenge website](https://claritychallenge.github.io/clarity_CC_doc/). You can contact the Clarity Team by email at [claritychallengecontact@gmail.com](claritychallengecontact@gmail.com).

In this repository, you will find code to support all Clarity Challenges, including baselines, tookkits, and systems from participants. **We encourage you to make your system/model open source and contribute to this repository.**

### The 2nd Clarity Enhancement Challenge (CEC2) has launched! [Take part](https://claritychallenge.github.io/clarity_CC_doc/docs/category/taking-part):fire::fire::fire:

## Installation
```bash
# First clone the repo
git clone https://github.com/claritychallenge/clarity.git
cd clarity

# Second create & activate environment with conda, see https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
conda create --name clarity python=3.8
conda activate clarity

# Third install requirements
pip install -r requirements.txt

# Last install with pip
pip install -e .
```

## Challenges
- [The 2nd Clarity Enhancement Challenge (CEC2)](./recipes/cec2)
- [The 1st Clarity Prediction Challenge (CPC1)](./recipes/cpc1)
- [The 1st Clarity Enhancement Challenge (CEC1)](./recipes/cec1/baseline)
