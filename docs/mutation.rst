Assignment & Mutation
=====================

.. automodule:: glom.mutation

.. contents:: Contents
   :local:

Assignment
----------

Deeply assign within an existing structure, given a path and a value.

.. autofunction:: glom.assign

.. autoclass:: glom.Assign

Updating
--------

Deeply update each value found at a path, computing every new value
from its own old value.

.. autofunction:: glom.update

.. autoclass:: glom.Update

Deletion
--------

Delete attributes from objects and keys from containers.

.. autofunction:: glom.delete

.. autoclass:: glom.Delete


Exceptions
----------

.. autoclass:: glom.PathAssignError

.. autoclass:: glom.PathDeleteError
