Event Data API
==============

EventData
---------
`databallpy.schemas.EventData`

.. autoclass:: databallpy.schemas.EventData
    :members:
    :show-inheritance:
    :no-index:


SPADL vocabulary
----------------

The ids of the SPADL actions returned by `EventData.to_spadl` and `Game.to_spadl`
are the indices in these lists.

.. autodata:: databallpy.utils.constants.SPADL_ACTION_TYPES
    :no-index:

.. autodata:: databallpy.utils.constants.SPADL_RESULTS
    :no-index:

.. autodata:: databallpy.utils.constants.SPADL_BODYPARTS
    :no-index:


XML Export
----------

`databallpy.events_to_xml`

.. autofunction:: databallpy.events_to_xml

`databallpy.Event`

.. autoclass:: databallpy.Event
    :members:
    :no-index:

`databallpy.LabelDict`

.. autoclass:: databallpy.LabelDict
    :members:
    :no-index:
