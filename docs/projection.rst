Writable Projections
====================

When glom projects an internal configuration object into an editable
form, the application usually needs a second, hand-written mapping to
apply edits back to the configuration. Keeping two mappings in sync is
error-prone.

:mod:`glom.projection` makes the read and the write-back explicit in a
single spec. Each form :class:`~glom.projection.Field` declares its
source path plus an optional forward/reverse transform pair.

.. automodule:: glom.projection

.. contents:: Contents
   :local:

Fields
------

.. autoclass:: glom.projection.Field

Projections
-----------

.. autoclass:: glom.projection.Project

Reading and writing convenience wrappers:

.. autofunction:: glom.projection.load

.. autofunction:: glom.projection.dump

Exceptions
----------

.. autoclass:: glom.projection.ProjectionError
