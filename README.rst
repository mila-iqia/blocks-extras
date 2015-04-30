Blocks extras
=============

The ``blocks-extras`` repository contains a variety of useful extensions to
Blocks_. The goal of this repository is for people to share useful extensions,
bricks, and other code snippets that are perhaps not general enough to go into
the core Blocks repository.

.. _Blocks: https://github.com/bartvm/blocks

Usage
-----

``blocks-extras`` is a namespace package, which means that everything can be
imported from ``blocks.extras``. For example, you can use:

.. code-block:: python

   from blocks.extras.extensions.plotting import Plot

Installation
------------

Clone to a directory of your choice.

.. code-block:: bash

   $ git clone git@github.com:mila-udem/blocks-extras.git

Install in editible mode using this command from within the folder you just
cloned:

.. code-block:: bash

   $ pip install -e .

If you want to update to the latest version, simply pull the latest
changes from GitHub.

.. code-block:: bash

   $ git pull
