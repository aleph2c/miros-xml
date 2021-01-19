.. Documentation master file
  Created:
  Authors: Scott Volk/Jessica Rush
  This will be the documentation's landing page
  It should at least contain the root `toctree` directive

.. meta::
  :description: miros-xml documentation
  :keywords: python statecharts, statecharts, miros, miros-xml, parallel regions
  :author: Scott Volk
  :copyright: Scott Volk
  :robots: index, follow

Miros-XML
=========

The miros-xml is a Python library which: consumes XML files describing a
statechart and turns it into a working python program:

* The parallel regions feature will be supported.

* Its XML mapping is inspired by the SCXML standard.

The majority of this document is written to describe how to implement parallel
regions using the miros event processor.

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   introduction
   quickstart
   techniques
   how_it_works
