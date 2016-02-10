Blocks extras
=============

The ``blocks-extras`` repository contains a variety of useful extensions to
Blocks_. The goal of this repository is for people to share useful extensions,
bricks, and other code snippets that are perhaps not general enough to go into
the core Blocks repository.

.. _Blocks: https://github.com/bartvm/blocks

Installation
------------

Clone to a directory of your choice.

.. code-block:: bash

   $ git clone git@github.com:mila-udem/blocks-extras.git

Usage
-----

.. code-block:: python

   from blocks_extras.extensions.plotting import Plot

A Note about ``Plot``
---------------------
Due to significant architectural changes upstream in Bokeh_, the `Plot` extension
is currently incompatible with Bokeh ≥ 0.11. A reimagined ``Plot`` extension will
probably require some sort of data storage backend of its own. Please see
``blocks-extras`` issue `#35`_ if you are interested in helping move this situation
forward. Until then, please use Bokeh 0.10 if you are interested in using the `Plot`
extension.

.. _Bokeh: http://bokeh.pydata.org/
.. _#35: http://github.com/mila-udem/blocks-extras/issues/35
