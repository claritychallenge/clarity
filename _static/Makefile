# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
SPHINXAPIBUILD = sphinx-apidoc
RECIPEDIR     = "$(SOURCEDIR)/../recipes"
# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	echo "Copying recipe README.md files for inclusion."
	rsync -av --exclude-from="recipe_rsync_exclude.txt" "$(RECIPEDIR)" "."
	@$(SPHINXAPIBUILD) --separate -o "$(SOURCEDIR)" "../" "../*setup*" "../*tests*" "../*.ipynb"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
