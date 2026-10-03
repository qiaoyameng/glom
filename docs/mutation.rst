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

Leaf-wise update
----------------

Update each destination at a path from its own old value, with access
to the original target and the call arguments.

.. autofunction:: glom.update

.. autoclass:: glom.Update

   .. autoattribute:: glom.mutation.UPDATE_TARGET

   .. autoattribute:: glom.mutation.UPDATE_ARGS

Deletion
--------

Delete attributes from objects and keys from containers.

.. autofunction:: glom.delete

.. autoclass:: glom.Delete


Exceptions
----------

.. autoclass:: glom.PathAssignError

.. autoclass:: glom.PathUpdateError

.. autoclass:: glom.PathDeleteError
