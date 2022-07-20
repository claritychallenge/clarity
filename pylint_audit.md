# Pylint

Initial run of pylint over complete code base. Created to help prioritise the fixes.

Pylint can be asked to report a specific error with e.g.,

`pylint --disable=all -e E1101 clarity`

Code currently rate 6.55

## Error - all done

150  E1101 - : Module 'torch' has no 'tensor' member;
1  E1120

## Warning - all done

27  W1203 - Use lazy % formatting in logging functions
14  W1514
7  W0612
6  W0511
5  W0622
4  W0707
4  W0613
4  W0221 - Variadics removed in overridden 'System.validation_step' method (arguments-differ)
3  W0621
2  W0404
2  W0105
1  W1510
1  W1114
1  W0702
1  W0235
1  W0201

## Convention

777  C0103 - variable names
49  C0116 - Missing function or method docstring (missing-function-docstring)
41  C0301 - Line too long (107/100) (line-too-long)
25  C0114 - missing-module-docstring
17  C0115 - missing-class-docstring
3  C0415 - Import outside toplevel (json) (import-outside-toplevel)
3  C0209 - consider using f-string
3  C0200 - consider using enumerate
3  C0123 - unidiomatic type check

## Refactor

39  R0801 - duplicate code
35  R0914 - Too many local variables
35  R0913 -  Too many arguments (6/5) (too-many-arguments)
17  R1725 - super-with-arguments
10  R1732 - consider using with
9  R0902 - too many instance attributes
5  R0915
4  R0201
3  R1714
3  R0903
3  R1705
3  R0402
2  R1735
2  R1731
1  R1730
1  R1724
1  R1723
1  R1721
1  R0912
1  R0901
1  R0205
