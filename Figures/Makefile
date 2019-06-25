SVG_FILES := $(wildcard *.svg)

all: svg

svg: $(SVG_FILES:%.svg=%.pdf)
	@echo Created figures from SVG files

SHELL=/bin/bash

#
# SVG-to-PDF
#

%.pdf: %.svg
	@echo Processing $@...
	@# The `abspath` function is required because Inkscape on Mac OS X cannot handle
	@# relative paths correctly.
	@inkscape -A "$(abspath $@)" "$(abspath $<)"
