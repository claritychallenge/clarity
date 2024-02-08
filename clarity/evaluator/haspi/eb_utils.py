"""
Module that contains the constants and functions
for the HAAQI, HASQI and HASPI models.

When the sample rate is 24000 Hz, the constants are:
    - COMPRESS_BASILAR_MEMBRANE_COEFS
    - MIDDLE_EAR_COEF
    - DELAY_COEFS
"""
COMPRESS_BASILAR_MEMBRANE_COEFS = {
    "24000": {
        "b": [0.09510798340249643, 0.09510798340249643],
        "a": [1.0, -0.8097840331950071],
    }
}
# Middle ear filter coefficients
MIDDLE_EAR_COEF = {
    "24000": {
        "butterworth_low_pass": [0.4341737512063021, 0.4341737512063021],
        "low_pass": [1.0, -0.13165249758739583],
        "butterworth_high_pass": [
            0.9372603902698923,
            -1.8745207805397845,
            0.9372603902698923,
        ],
        "high_pass": [1.0, -1.8705806407352794, 0.8784609203442912],
    }
}

DELAY_COEFS = [
    0,
    50,
    92,
    127,
    157,
    183,
    205,
    225,
    242,
    256,
    267,
    275,
    283,
    291,
    299,
    305,
    311,
    316,
    320,
    325,
    329,
    332,
    335,
    338,
    340,
    341,
    342,
    344,
    344,
    345,
    346,
    347,
]
