{# docs/_templates/autosummary/class.rst #}
{{ name }}
{{ "=" * name|length }}

.. autoclass:: {{ fullname }}
   :noindex:
   :members:       {# Typically you want to list members for classes #}
   :undoc-members:
   :show-inheritance: