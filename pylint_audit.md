# Pylint

Initial run of pylint over complete code base. Created to help prioritise the fixes.

Pylint can be asked to report a specific error with e.g.,

`pylint --disable=all -e E1101 clarity`

Code currently rate 6.55

## Warning - all done

6  W0511 - TODO messages in code
4  W0221 - Variadics removed in overridden 'System.validation_step' method (arguments-differ)
1  W0703 - Catching too general exception Exception (broad-except)

## Convention

49  C0116 - Missing function or method docstring (missing-function-docstring)
25  C0114 - missing-module-docstring
17  C0115 - missing-class-docstring
1  C0200 - consider using enumerate
1  C0301 - line too long (is an URL so awkward to split)

## Refactor

39  R0801 - duplicate code (!!)
35  R0914 - Too many local variables
15  R0913 - Too many arguments
9  R0902 - too many instance attributes
5  R0915 - Too many statements
4  R0201 - no self use
3  R0402 - consider using from import
3  R0903 - too few public methods
2  R1705 - no else return
1  R0901 - too many ancestors
1  R0912 - too many branches
1  R1721 - consider swap variables

## Duplicated code

### fir2 and firwin2

==clarity.evaluator.msbg.msbg_utils:[331:415]
==clarity.enhancer.nalr:[20:100]
functions fir2 and firwin2 - identical

### numpy array constnatns in msbg

==clarity.evaluator.msbg.msbg_utils:[116:166]
==clarity.predictor.torch_msbg:[271:315]
large constant np.arrays
FF_ED - free field to ear drum array definition
midear - constant np.array
hz - centre frequencies
df_ed
ITU_Hz
ITU_erp_drp

### functions for reading and writing signals

==clarity.data.scene_renderer_cec1:[50:85]
==clarity.enhancer.gha.gha_interface:[178:215]
function read_signal,

==clarity.data.scene_renderer_cec1:[92:102]
==clarity.enhancer.gha.gha_interface:[218:228]
function read_signal - identical
function write_signal - identical accept for minor refactor to handle sample rate

- read and write signal also in msbg_utils - but a bit different

### Audiogram dataclass

==clarity.enhancer.gha.audiogram:[35:43]
==clarity.evaluator.msbg.audiogram:[92:100]
class Audiogram

- binaural audiogram in enhancer/gha
- monaural audiogram in evaluator/msbg

 Very similar - e.g. shared code in severity() function

 Elsewhere audiograms are passed around just as separate left ear loss, right ear loss and cfs parameters

### functions fir2 and firwin2

==clarity.evaluator.msbg.msbg_utils:[331:415]
==clarity.enhancer.nalr:[20:100]

### msbg constant arrays

==clarity.evaluator.msbg.msbg_utils:[116:166]
==clarity.predictor.torch_msbg:[271:315]

large constant np.arrays
FF_ED - free field to ear drum array definition
midear - constant np.array
hz - centre frequencies
df_ed
ITU_Hz
ITU_erp_drp

In general, much of torch_msbg is a re-write of msbg.

### code for reading and writing signals

==clarity.data.scene_renderer_cec1:[50:85]
==clarity.enhancer.gha.gha_interface:[178:215]
function read_signal,

==clarity.data.scene_renderer_cec1:[92:102]
==clarity.enhancer.gha.gha_interface:[218:228]
function read_signal - identical
function write_signal - identical accept for minor refactor to handle sample rate

- read and write signal also in msbg_utils - but a bit different

### Handling of audiograms

==clarity.enhancer.gha.audiogram:[35:43]
==clarity.evaluator.msbg.audiogram:[92:100]

dataclasses for Audiogram

- binaural audiogram in enhancer/gha
- monaural audiogram in evaluator/msbg
 Very similar - e.g. shared code in severity() function

Elsewhere audiograms are passed around just as separate left ear loss, right ear loss and cfs parameters

## Specific warnings

************* Module clarity
clarity/__init__.py:1:0: C0114: Missing module docstring (missing-module-docstring)
************* Module clarity.evaluator.msbg.smearing
clarity/evaluator/msbg/smearing.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/evaluator/msbg/smearing.py:85:0: R0914: Too many local variables (24/15) (too-many-locals)
clarity/evaluator/msbg/smearing.py:154:0: C0115: Missing class docstring (missing-class-docstring)
clarity/evaluator/msbg/smearing.py:167:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/evaluator/msbg/smearing.py:154:0: R0903: Too few public methods (1/2) (too-few-public-methods)
************* Module clarity.evaluator.msbg.msbg_utils
clarity/evaluator/msbg/msbg_utils.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/evaluator/msbg/msbg_utils.py:351:0: R0914: Too many local variables (19/15) (too-many-locals)
clarity/evaluator/msbg/msbg_utils.py:416:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/evaluator/msbg/msbg_utils.py:424:0: R0914: Too many local variables (20/15) (too-many-locals)
clarity/evaluator/msbg/msbg_utils.py:495:0: R0914: Too many local variables (24/15) (too-many-locals)
************* Module clarity.evaluator.msbg.cochlea
clarity/evaluator/msbg/cochlea.py:10:1: W0511: TODO: Fix power overflow error when (expansion_ratios[ixch] - 1) < 0 (fixme)
clarity/evaluator/msbg/cochlea.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/evaluator/msbg/cochlea.py:72:0: R0913: Too many arguments (8/7) (too-many-arguments)
clarity/evaluator/msbg/cochlea.py:72:0: R0914: Too many local variables (16/15) (too-many-locals)
clarity/evaluator/msbg/cochlea.py:181:0: R0903: Too few public methods (1/2) (too-few-public-methods)
************* Module clarity.evaluator.msbg.audiogram
clarity/evaluator/msbg/audiogram.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/evaluator/msbg/audiogram.py:8:0: C0116: Missing function or method docstring (missing-function-docstring)
************* Module clarity.evaluator.msbg.msbg
clarity/evaluator/msbg/msbg.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/evaluator/msbg/msbg.py:43:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/evaluator/msbg/msbg.py:153:4: R0914: Too many local variables (18/15) (too-many-locals)
************* Module clarity.evaluator.mbstoi.mbstoi
clarity/evaluator/mbstoi/mbstoi.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/evaluator/mbstoi/mbstoi.py:15:0: R0914: Too many local variables (63/15) (too-many-locals)
clarity/evaluator/mbstoi/mbstoi.py:15:0: R0915: Too many statements (92/50) (too-many-statements)
************* Module clarity.evaluator.mbstoi
clarity/evaluator/mbstoi/__init__.py:1:0: C0114: Missing module docstring (missing-module-docstring)
************* Module clarity.evaluator.mbstoi.mbstoi_utils
clarity/evaluator/mbstoi/mbstoi_utils.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/evaluator/mbstoi/mbstoi_utils.py:9:0: R0913: Too many arguments (16/7) (too-many-arguments)
clarity/evaluator/mbstoi/mbstoi_utils.py:9:0: R0914: Too many local variables (44/15) (too-many-locals)
clarity/evaluator/mbstoi/mbstoi_utils.py:9:0: R0915: Too many statements (56/50) (too-many-statements)
clarity/evaluator/mbstoi/mbstoi_utils.py:153:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/evaluator/mbstoi/mbstoi_utils.py:169:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/evaluator/mbstoi/mbstoi_utils.py:184:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/evaluator/mbstoi/mbstoi_utils.py:197:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/evaluator/mbstoi/mbstoi_utils.py:236:0: R0914: Too many local variables (23/15) (too-many-locals)
************* Module clarity.evaluator.haspi.eb
clarity/evaluator/haspi/eb.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/evaluator/haspi/eb.py:6:0: R0914: Too many local variables (58/15) (too-many-locals)
clarity/evaluator/haspi/eb.py:6:0: R0915: Too many statements (60/50) (too-many-statements)
clarity/evaluator/haspi/eb.py:309:0: R0914: Too many local variables (18/15) (too-many-locals)
clarity/evaluator/haspi/eb.py:373:0: R0914: Too many local variables (17/15) (too-many-locals)
clarity/evaluator/haspi/eb.py:471:0: R0914: Too many local variables (30/15) (too-many-locals)
clarity/evaluator/haspi/eb.py:565:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/evaluator/haspi/eb.py:618:0: R0913: Too many arguments (8/7) (too-many-arguments)
clarity/evaluator/haspi/eb.py:618:0: R0914: Too many local variables (16/15) (too-many-locals)
clarity/evaluator/haspi/eb.py:751:0: R0914: Too many local variables (33/15) (too-many-locals)
clarity/evaluator/haspi/eb.py:860:0: R0914: Too many local variables (25/15) (too-many-locals)
************* Module clarity.evaluator.haspi.ebm
clarity/evaluator/haspi/ebm.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/evaluator/haspi/ebm.py:7:0: R0914: Too many local variables (19/15) (too-many-locals)
clarity/evaluator/haspi/ebm.py:72:0: R0914: Too many local variables (20/15) (too-many-locals)
clarity/evaluator/haspi/ebm.py:173:0: R0914: Too many local variables (32/15) (too-many-locals)
************* Module clarity.evaluator.haspi
clarity/evaluator/haspi/__init__.py:1:0: C0114: Missing module docstring (missing-module-docstring)
************* Module clarity.evaluator.haspi.haspi
clarity/evaluator/haspi/haspi.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/evaluator/haspi/haspi.py:4:0: R0914: Too many local variables (30/15) (too-many-locals)
clarity/evaluator/haspi/haspi.py:79:0: R0913: Too many arguments (9/7) (too-many-arguments)
************* Module clarity.evaluator.haspi.ip
clarity/evaluator/haspi/ip.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/evaluator/haspi/ip.py:245:0: R0914: Too many local variables (18/15) (too-many-locals)
************* Module clarity.enhancer.nalr
clarity/enhancer/nalr.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/enhancer/nalr.py:40:0: R0914: Too many local variables (19/15) (too-many-locals)
clarity/enhancer/nalr.py:101:0: C0115: Missing class docstring (missing-class-docstring)
clarity/enhancer/nalr.py:119:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/enhancer/nalr.py:163:4: R0201: Method could be a function (no-self-use)
************* Module clarity.enhancer.compressor
clarity/enhancer/compressor.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/enhancer/compressor.py:4:0: C0115: Missing class docstring (missing-class-docstring)
clarity/enhancer/compressor.py:4:0: R0902: Too many instance attributes (10/7) (too-many-instance-attributes)
clarity/enhancer/compressor.py:5:4: R0913: Too many arguments (8/7) (too-many-arguments)
clarity/enhancer/compressor.py:28:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/enhancer/compressor.py:33:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/enhancer/compressor.py:38:4: C0116: Missing function or method docstring (missing-function-docstring)
************* Module clarity.enhancer.gha.gainrule_camfit
clarity/enhancer/gha/gainrule_camfit.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/enhancer/gha/gainrule_camfit.py:193:0: R0914: Too many local variables (20/15) (too-many-locals)
clarity/enhancer/gha/gainrule_camfit.py:240:11: W0703: Catching too general exception Exception (broad-except)
clarity/enhancer/gha/gainrule_camfit.py:307:0: R0914: Too many local variables (41/15) (too-many-locals)
clarity/enhancer/gha/gainrule_camfit.py:307:0: R0915: Too many statements (57/50) (too-many-statements)
************* Module clarity.enhancer.gha.gha_utils
clarity/enhancer/gha/gha_utils.py:1:0: C0114: Missing module docstring (missing-module-docstring)
************* Module clarity.enhancer.gha.gha_interface
clarity/enhancer/gha/gha_interface.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/enhancer/gha/gha_interface.py:15:0: C0115: Missing class docstring (missing-class-docstring)
clarity/enhancer/gha/gha_interface.py:15:0: R0902: Too many instance attributes (10/7) (too-many-instance-attributes)
clarity/enhancer/gha/gha_interface.py:16:4: R0913: Too many arguments (11/7) (too-many-arguments)
************* Module clarity.enhancer.gha.audiogram
clarity/enhancer/gha/audiogram.py:1:0: C0114: Missing module docstring (missing-module-docstring)
************* Module clarity.enhancer.dnn.mc_conv_tasnet
clarity/enhancer/dnn/mc_conv_tasnet.py:349:9: W0511: TODO: when P = 3 here works fine, but when P = 2 maybe need to pad? (fixme)
clarity/enhancer/dnn/mc_conv_tasnet.py:432:1: W0511: TODO: Use nn.LayerNorm to impl cLN to speed up (fixme)
clarity/enhancer/dnn/mc_conv_tasnet.py:479:9: W0511: TODO: in torch 1.0, torch.mean() support dim list (fixme)
clarity/enhancer/dnn/mc_conv_tasnet.py:32:0: C0301: Line too long (123/100) (line-too-long)
clarity/enhancer/dnn/mc_conv_tasnet.py:7:0: R0402: Use 'from torch import nn' instead (consider-using-from-import)
clarity/enhancer/dnn/mc_conv_tasnet.py:59:0: C0115: Missing class docstring (missing-class-docstring)
clarity/enhancer/dnn/mc_conv_tasnet.py:59:0: R0902: Too many instance attributes (17/7) (too-many-instance-attributes)
clarity/enhancer/dnn/mc_conv_tasnet.py:60:4: R0913: Too many arguments (14/7) (too-many-arguments)
clarity/enhancer/dnn/mc_conv_tasnet.py:193:0: C0115: Missing class docstring (missing-class-docstring)
clarity/enhancer/dnn/mc_conv_tasnet.py:222:0: C0115: Missing class docstring (missing-class-docstring)
clarity/enhancer/dnn/mc_conv_tasnet.py:223:4: R0913: Too many arguments (12/7) (too-many-arguments)
clarity/enhancer/dnn/mc_conv_tasnet.py:223:4: R0914: Too many local variables (22/15) (too-many-locals)
clarity/enhancer/dnn/mc_conv_tasnet.py:309:0: C0115: Missing class docstring (missing-class-docstring)
clarity/enhancer/dnn/mc_conv_tasnet.py:310:4: R0913: Too many arguments (9/7) (too-many-arguments)
clarity/enhancer/dnn/mc_conv_tasnet.py:354:0: C0115: Missing class docstring (missing-class-docstring)
clarity/enhancer/dnn/mc_conv_tasnet.py:355:4: R0913: Too many arguments (9/7) (too-many-arguments)
clarity/enhancer/dnn/mc_conv_tasnet.py:442:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/enhancer/dnn/mc_conv_tasnet.py:468:4: C0116: Missing function or method docstring (missing-function-docstring)
************* Module clarity.enhancer.dsp.filter
clarity/enhancer/dsp/filter.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/enhancer/dsp/filter.py:3:0: R0402: Use 'from torch import nn' instead (consider-using-from-import)
clarity/enhancer/dsp/filter.py:9:0: C0115: Missing class docstring (missing-class-docstring)
clarity/enhancer/dsp/filter.py:51:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/enhancer/dsp/filter.py:51:4: R0914: Too many local variables (19/15) (too-many-locals)
************* Module clarity.predictor.torch_stoi
clarity/predictor/torch_stoi.py:18:0: R0902: Too many instance attributes (12/7) (too-many-instance-attributes)
clarity/predictor/torch_stoi.py:91:4: R0914: Too many local variables (23/15) (too-many-locals)
clarity/predictor/torch_stoi.py:208:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/predictor/torch_stoi.py:243:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/predictor/torch_stoi.py:249:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/predictor/torch_stoi.py:257:0: C0116: Missing function or method docstring (missing-function-docstring)
************* Module clarity.predictor.torch_msbg
clarity/predictor/torch_msbg.py:9:0: R0402: Use 'from torch import nn' instead (consider-using-from-import)
clarity/predictor/torch_msbg.py:34:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/predictor/torch_msbg.py:34:0: R0914: Too many local variables (16/15) (too-many-locals)
clarity/predictor/torch_msbg.py:95:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/predictor/torch_msbg.py:145:0: C0115: Missing class docstring (missing-class-docstring)
clarity/predictor/torch_msbg.py:145:0: R0902: Too many instance attributes (33/7) (too-many-instance-attributes)
clarity/predictor/torch_msbg.py:146:4: R0914: Too many local variables (77/15) (too-many-locals)
clarity/predictor/torch_msbg.py:146:4: R0912: Too many branches (17/12) (too-many-branches)
clarity/predictor/torch_msbg.py:146:4: R0915: Too many statements (150/50) (too-many-statements)
clarity/predictor/torch_msbg.py:648:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/predictor/torch_msbg.py:676:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/predictor/torch_msbg.py:685:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/predictor/torch_msbg.py:729:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/predictor/torch_msbg.py:771:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/predictor/torch_msbg.py:802:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/predictor/torch_msbg.py:813:0: C0115: Missing class docstring (missing-class-docstring)
clarity/predictor/torch_msbg.py:813:0: R0902: Too many instance attributes (11/7) (too-many-instance-attributes)
clarity/predictor/torch_msbg.py:814:4: R0914: Too many local variables (16/15) (too-many-locals)
clarity/predictor/torch_msbg.py:864:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/predictor/torch_msbg.py:869:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/predictor/torch_msbg.py:892:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/predictor/torch_msbg.py:897:4: C0116: Missing function or method docstring (missing-function-docstring)
************* Module clarity.data.scene_renderer_cec1
clarity/data/scene_renderer_cec1.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/data/scene_renderer_cec1.py:14:0: R0902: Too many instance attributes (9/7) (too-many-instance-attributes)
clarity/data/scene_renderer_cec1.py:21:4: R0913: Too many arguments (10/7) (too-many-arguments)
clarity/data/scene_renderer_cec1.py:161:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/data/scene_renderer_cec1.py:161:4: R0913: Too many arguments (11/7) (too-many-arguments)
clarity/data/scene_renderer_cec1.py:161:4: R0914: Too many local variables (32/15) (too-many-locals)
clarity/data/scene_renderer_cec1.py:302:0: C0116: Missing function or method docstring (missing-function-docstring)
************* Module clarity.data.demo_data
clarity/data/demo_data.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/data/demo_data.py:10:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/data/demo_data.py:19:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/data/demo_data.py:24:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/data/demo_data.py:29:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/data/demo_data.py:34:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/data/demo_data.py:39:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/data/demo_data.py:44:0: C0116: Missing function or method docstring (missing-function-docstring)
************* Module clarity.data.utils
clarity/data/utils.py:1:0: C0114: Missing module docstring (missing-module-docstring)
************* Module clarity.data.scene_builder_cec2
clarity/data/scene_builder_cec2.py:35:0: C0116: Missing function or method docstring (missing-function-docstring)
clarity/data/scene_builder_cec2.py:277:0: R0914: Too many local variables (17/15) (too-many-locals)
clarity/data/scene_builder_cec2.py:346:0: R0914: Too many local variables (23/15) (too-many-locals)
clarity/data/scene_builder_cec2.py:469:0: R0902: Too many instance attributes (8/7) (too-many-instance-attributes)
clarity/data/scene_builder_cec2.py:472:4: R0913: Too many arguments (8/7) (too-many-arguments)
clarity/data/scene_builder_cec2.py:493:0: R1721: Unnecessary use of a comprehension, use self.scenes instead. (unnecessary-comprehension)
clarity/data/scene_builder_cec2.py:500:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/data/scene_builder_cec2.py:639:4: R0913: Too many arguments (9/7) (too-many-arguments)
************* Module clarity.data.scene_renderer_cec2
clarity/data/scene_renderer_cec2.py:155:9: W0511: TODO: The delay does not appear to correctly align the signals as expected (fixme)
clarity/data/scene_renderer_cec2.py:219:9: W0511: TODO: set target to a fixed reference level?? (fixme)
clarity/data/scene_renderer_cec2.py:65:0: R0902: Too many instance attributes (8/7) (too-many-instance-attributes)
clarity/data/scene_renderer_cec2.py:203:4: R0914: Too many local variables (22/15) (too-many-locals)
clarity/data/scene_renderer_cec2.py:282:4: R0201: Method could be a function (no-self-use)
clarity/data/scene_renderer_cec2.py:297:4: R0914: Too many local variables (27/15) (too-many-locals)
************* Module clarity.data.HOA_tools_cec2
clarity/data/HOA_tools_cec2.py:120:4: R1705: Unnecessary "elif" after "return" (no-else-return)
clarity/data/HOA_tools_cec2.py:147:4: R1705: Unnecessary "elif" after "return" (no-else-return)
clarity/data/HOA_tools_cec2.py:241:0: R0903: Too few public methods (1/2) (too-few-public-methods)
************* Module clarity.engine.system
clarity/engine/system.py:9:0: C0115: Missing class docstring (missing-class-docstring)
clarity/engine/system.py:9:0: R0901: Too many ancestors (8/7) (too-many-ancestors)
clarity/engine/system.py:10:4: R0913: Too many arguments (8/7) (too-many-arguments)
clarity/engine/system.py:62:4: W0221: Number of parameters was 3 in 'LightningModule.training_step' and is now 3 in overridden 'System.training_step' method (arguments-differ)
clarity/engine/system.py:62:4: W0221: Variadics removed in overridden 'System.training_step' method (arguments-differ)
clarity/engine/system.py:76:4: W0221: Number of parameters was 3 in 'LightningModule.validation_step' and is now 3 in overridden 'System.validation_step' method (arguments-differ)
clarity/engine/system.py:76:4: W0221: Variadics removed in overridden 'System.validation_step' method (arguments-differ)
************* Module clarity.engine.losses
clarity/engine/losses.py:1:0: C0114: Missing module docstring (missing-module-docstring)
clarity/engine/losses.py:6:0: C0115: Missing class docstring (missing-class-docstring)
clarity/engine/losses.py:10:4: R0201: Method could be a function (no-self-use)
clarity/engine/losses.py:35:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/engine/losses.py:39:0: C0115: Missing class docstring (missing-class-docstring)
clarity/engine/losses.py:44:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/engine/losses.py:44:4: R0201: Method could be a function (no-self-use)
clarity/engine/losses.py:47:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/engine/losses.py:59:0: C0115: Missing class docstring (missing-class-docstring)
clarity/engine/losses.py:64:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/engine/losses.py:68:0: C0115: Missing class docstring (missing-class-docstring)
clarity/engine/losses.py:85:4: C0116: Missing function or method docstring (missing-function-docstring)
clarity/engine/losses.py:107:4: C0116: Missing function or method docstring (missing-function-docstring)

-------------------------------------------------------------------
Your code has been rated at 9.44/10 (previous run: 10.00/10, -0.56)
