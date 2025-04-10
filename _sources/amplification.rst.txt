Amplification
=============

NAL-R
-----

Example:

.. code-block:: python

   from clarity.enhancer.nalr import NALR

   nalr = NALR(nfir=220, sample_rate=44100.0)
   fir = nalr.build(audiogram)
   enhanced_signal = nalr.apply(fir, wav_signal)


.. autoclass:: clarity.enhancer.nalr.NALR
   :members:
   :undoc-members:
   :show-inheritance:

Multiband Dynamic Range Compressor
----------------------------------

Compressor
^^^^^^^^^^

.. autoclass:: clarity.enhancer.multiband_compressor.compressor_qmul.Compressor
   :members:
   :undoc-members:

Crossover
^^^^^^^^^

.. autoclass:: clarity.enhancer.multiband_compressor.crossover.Crossover
   :members:

MultibandCompressor
^^^^^^^^^^^^^^^^^^^

.. autoclass:: clarity.enhancer.multiband_compressor.multiband_compressor.MultibandCompressor
   :members: