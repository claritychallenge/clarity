# Baseline Recipes for the ICASSP grand Challenge

## Contents
```text
baseline
└───config.yaml - parameters for recipes
└───enhance.py - enhancement script. Identical to CEC2 baseline
└───evaluate.py  - runs combined HASPI/HASQI evaluation
└───__init__.py
└───README.md
```

To run, override path.root when executing to point to audio file directory

<code>
$> python evaluate.py path.root='/path/to/audio'
</code>