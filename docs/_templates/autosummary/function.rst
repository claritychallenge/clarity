{{ name }}
{{ "=" * name|length }}  {# Use the short name for the heading and underline #}

.. autofunction:: {{ fullname }}
   :noindex:
   {# Add any specific autodoc options you need for this function here,
      or rely on 'autodoc_default_options' in conf.py
   #}