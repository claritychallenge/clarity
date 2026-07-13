{# docs/_templates/autosummary/module.rst #}
{{ name }}
{{ "=" * name|length }}

.. automodule:: {{ fullname }}
   :noindex:
   :members:       {# You'll typically want to show members for modules #}
   :undoc-members:
   :show-inheritance: