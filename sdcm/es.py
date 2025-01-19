import logging
from functools import cached_property

import elasticsearch

from sdcm.keystore import KeyStore

LOGGER = logging.getLogger(__name__)


class ES(elasticsearch.Elasticsearch):
    """
    Provides interface for Elasticsearch DB
    """

    def __init__(self):
        super().__init__(**self.conf)

    @cached_property
    def conf(self):
        conf_dict = KeyStore().get_elasticsearch_token()
        conf_dict.pop('kibana_url', None)
        return conf_dict

    def _create_index(self, index):
        self.indices.create(index=index, ignore=400)  # pylint: disable=unexpected-keyword-arg

    def create_doc(self, index, doc_type, doc_id, body):
        """
        Add document in json format
        """
        LOGGER.info('Create doc')
        LOGGER.info('INDEX: %s', index)
        LOGGER.info('DOC_TYPE: %s', doc_type)
        LOGGER.info('DOC_ID: %s', doc_id)
        LOGGER.info('BODY: %s', body)
        self._create_index(index)
        if self.exists(index=index, doc_type=doc_type, id=doc_id):
            self.update(index=index, doc_type=doc_type, id=doc_id, body={'doc': body})
        else:
            self.create(index=index, doc_type=doc_type, id=doc_id, body=body)

    def update_doc(self, index, doc_type, doc_id, body):
        """
        Update document with partial data
        """
        LOGGER.info('Update doc %s with info %s', doc_id, body)
        self.update(index=index, doc_type=doc_type, id=doc_id, body=dict(doc=body))

    def get_all(self, index, limit=1000):
        """
        Search for documents for the certain index
        """
        return self.search(index=index, size=limit)  # pylint: disable=unexpected-keyword-arg

    def get_doc(self, index, doc_id, doc_type='_all'):
        """
        Get document by id
        """
        doc = self.get(index=index, doc_type=doc_type, id=doc_id, ignore=[  # pylint: disable=unexpected-keyword-arg
                       400, 404])
        if not doc['found']:
            LOGGER.warning('Document not found: %s %s', doc_id, doc_type)
            return None
        return doc

    def delete_doc(self, index, doc_type, doc_id):
        """
        Delete document
        """
        if self.get_doc(index, doc_id, doc_type):
            self.delete(index=index, doc_type=doc_type, id=doc_id)
