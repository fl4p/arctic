import logging
from collections import namedtuple

import pymongo
from pymongo.errors import OperationFailure

logger = logging.getLogger(__name__)

# PyMongo 4 REMOVED Database.authenticate(); credentials are supplied at MongoClient
# construction instead (see Arctic._conn). NB: on pymongo>=4 `db.authenticate` does NOT
# raise AttributeError -- Database.__getattr__ returns a Collection named "authenticate"
# (which is not callable) -- so we must gate on the driver version, not on getattr/hasattr.
_PYMONGO4 = int(pymongo.version.split('.', 1)[0]) >= 4


def authenticate(db, user, password):
    """
    Return True / False on authentication success.

    PyMongo 2.6 changed the auth API to raise on Auth failure.

    PyMongo 4 removed Database.authenticate(): the client connection is already
    authenticated (creds passed to MongoClient in Arctic._conn), so this is a no-op
    returning True. The pymongo<4 path is unchanged.
    """
    if _PYMONGO4:
        return True  # pymongo>=4: auth happens at client construction, not per-DB
    try:
        logger.debug("Authenticating {} with {}".format(db, user))
        return db.authenticate(user, password)
    except OperationFailure as e:
        logger.debug("Auth Error %s" % e)
    return False


Credential = namedtuple("MongoCredentials", ['database', 'user', 'password'])


def get_auth(host, app_name, database_name):
    """
    Authentication hook to allow plugging in custom authentication credential providers
    """
    from .hooks import _get_auth_hook
    return _get_auth_hook(host, app_name, database_name)
