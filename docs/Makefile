# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
USERGUIDEDIR  = userguide
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile


clean:
	@echo "[Makefile] cleaning..."
	@rm -rf "./$(BUILDDIR)"
	@rm -rf "./$(SOURCEDIR)/examples"
	@rm -rf "./$(SOURCEDIR)/generated_api"
	@rm -rf "./$(SOURCEDIR)/quickstart"
	@rm -rf "./$(SOURCEDIR)/userguide"
	@rm -rf "./$(SOURCEDIR)/README.rst"

# copy-examples:
# 	@echo "[Makefile] copying examples..."
# 	@cp -r ../examples "$(SOURCEDIR)/examples"
# 	@rm "$(SOURCEDIR)/examples/00-quickstart.ipynb"

# copy-quickstart:
# 	@echo "[Makefile] copying quickstart notebook..."
# 	@mkdir -p "$(SOURCEDIR)/quickstart"
# 	@cp -r ../examples/00-quickstart.ipynb "$(SOURCEDIR)/quickstart/"
# 	@cp -r ../examples/static "$(SOURCEDIR)/quickstart/"

generate-readme:
	@echo "[Makefile] generating README page..."
	@cp ../README.md "$(SOURCEDIR)"
	@sed -i '1s/^/# Home\n/' $(SOURCEDIR)/README.md

generate-api:
	@echo "[Makefile] generating API using sphinx-apidoc..."
	@sphinx-apidoc -e -f --templatedir templates -o "$(SOURCEDIR)/generated_api" ../oats

html:
	# Note: this target has to be called "html" because its name is given as an argument to Sphinx
	@echo "[Makefile] generating HTML pages using sphinx-build..."
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# build-all-docs: clean copy-examples copy-quickstart generate-readme generate-userguide generate-api html
build-all-docs: clean generate-readme generate-api html


# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
#%: Makefile
#	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)



