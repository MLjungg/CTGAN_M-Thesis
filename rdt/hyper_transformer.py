"""Hyper transformer module."""

import re
import warnings
from copy import deepcopy

import numpy as np

from rdt.transformers import (
    BooleanTransformer, CategoricalTransformer, DatetimeTransformer, LabelEncodingTransformer,
    NumericalTransformer, OneHotEncodingTransformer, load_transformers)


class HyperTransformer:
    """HyperTransformer class.

    The ``HyperTransformer`` class contains a collection of ``transformers`` that can be
    used to transform and reverse transform one or more columns at once.

    Args:
        transformers (dict or None):
            dict associating column names with transformers, which can be either passed
            directly as an instance or as a dict specification. If ``None``, a simple
            ``transformers`` dict is built automatically from the data.
        copy (bool):
            Whether to make a copy of the input data or not. Defaults to ``True``.
        anonymize (dict or None):
            Dictionary specifying the names and ``faker`` categories of the categorical
            columns that need to be anonymized. Defaults to ``None``.
        dtypes (list or None):
            List of column data types to use when building the ``transformers`` dict
            automatically. If not passed, the ``DataFrame.dtypes`` are used.
        dtype_transformers (dict or None):
            Transformer templates to use for each dtype. Passed as a dictionary of
            dtype kinds ('i', 'f', 'O', 'b', 'M') and transformer names, classes
            or instances.

    Example:
        Create a simple ``HyperTransformer`` instance that will decide which transformers
        to use based on the fit data ``dtypes``.

        >>> ht = HyperTransformer()

        Create a ``HyperTransformer`` passing a list of dtypes.

        >>> ht = HyperTransformer(dtypes=[int, 'object', np.float64, 'datetime', 'bool'])

        Create a ``HyperTransformer`` passing a ``transformers`` dict.

        >>> transformers = {
        ...     'a': NumericalTransformer(dtype=float),
        ...     'b': {
        ...         'class': 'NumericalTransformer',
        ...         'kwargs': {
        ...             'dtype': int
        ...         }
        ...     }
        ... }
        >>> ht = HyperTransformer(transformers)
    """

    _TRANSFORMER_TEMPLATES = {
        'numerical': NumericalTransformer,
        'integer': NumericalTransformer(dtype=int),
        'float': NumericalTransformer(dtype=float),
        'categorical': CategoricalTransformer,
        'categorical_fuzzy': CategoricalTransformer(fuzzy=True),
        'one_hot_encoding': OneHotEncodingTransformer(error_on_unknown=False),
        'label_encoding': LabelEncodingTransformer,
        'boolean': BooleanTransformer,
        'datetime': DatetimeTransformer,
    }
    _DTYPE_TRANSFORMERS = {
        'i': 'numerical',
        'f': 'numerical',
        'O': 'categorical',
        'b': 'boolean',
        'M': 'datetime',
    }

    def __init__(self, transformers=None, copy=True, anonymize=None,
                 dtypes=None, dtype_transformers=None):
        self.transformers = transformers
        self._transformers = dict()
        self.copy = copy
        self.anonymize = anonymize or dict()
        self.dtypes = dtypes
        self.dtype_transformers = self._DTYPE_TRANSFORMERS.copy()
        if dtype_transformers:
            self.dtype_transformers.update(dtype_transformers)

    def _analyze(self, data):
        """Build a ``dict`` with column names and transformers from a given ``pandas.DataFrame``.

        When ``self.dtypes`` is ``None``, use the dtypes from the input data.

        When ``dtype`` is:
            - ``int``: a ``NumericalTransformer`` is created with ``dtype=int``.
            - ``float``: a ``NumericalTransformer`` is created with ``dtype=float``.
            - ``object`` or ``category``: a ``CategoricalTransformer`` is created.
            - ``bool``: a ``BooleanTransformer`` is created.
            - ``datetime``: a ``DatetimeTransformer`` is created.

        Any other ``dtype`` is not supported and raises a ``ValueError``.

        Args:
            data (pandas.DataFrame):
                Data used to analyze the ``pandas.DataFrame`` dtypes.

        Returns:
            dict:
                Mapping of column names and transformer instances.

        Raises:
            ValueError:
                if a ``dtype`` is not supported by the `HyperTransformer``.
        """
        transformers = dict()
        if self.dtypes:
            dtypes = self.dtypes
        else:
            dtypes = [
                data[column].dropna().infer_objects().dtype
                for column in data.columns
            ]

        for name, dtype in zip(data.columns, dtypes):
            try:
                kind = np.dtype(dtype).kind
            except TypeError:
                # probably category
                kind = 'O'

            transformer_template = self.dtype_transformers[kind]
            if not transformer_template:
                raise ValueError('Unsupported dtype: {}'.format(dtype))

            if isinstance(transformer_template, str):
                transformer_template = self._TRANSFORMER_TEMPLATES[transformer_template]

            if not isinstance(transformer_template, type):
                transformer = deepcopy(transformer_template)
            elif self.anonymize and transformer_template == CategoricalTransformer:
                warnings.warn(
                    'Categorical anonymization is deprecated and will be removed from RDT soon.',
                    DeprecationWarning
                )
                transformer = CategoricalTransformer(anonymize=self.anonymize)
            else:
                transformer = transformer_template()

            transformers[name] = transformer

        return transformers

    def fit(self, data):
        """Fit the transformers to the data.

        Args:
            data (pandas.DataFrame):
                Data to fit the transformers to.
        """
        if self.transformers is not None:
            self._transformers = load_transformers(self.transformers)
        else:
            self._transformers = self._analyze(data)

        for column_name, transformer in self._transformers.items():
            column = data[column_name]
            transformer.fit(column)

    def transform(self, data):
        """Transform the data.

        If ``self.copy`` is ``True`` make a copy of the input data to avoid modifying it.

        Args:
            data (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        self.column_names = []
        if self.copy:
            data = data.copy()

        for column_name, transformer in self._transformers.items():
            column = data.pop(column_name)
            transformed = transformer.transform(column)
            self.column_names.append(column_name)

            shape = transformed.shape

            if len(shape) == 2:
                for index in range(shape[1]):
                    new_column = '{}#{}'.format(column_name, index)
                    data[new_column] = transformed[:, index]

            else:
                data[column_name] = transformed

        return data

    def fit_transform(self, data):
        """Fit the transformers to the data and then transform it.

        Args:
            data (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        self.fit(data)
        return self.transform(data)

    @staticmethod
    def _get_columns(data, column_name):
        """Get one or more columns that match a given name.

        Args:
            data (pandas.DataFrame):
                Table to perform the matching.
            column_name (str):
                Name to match the columns.

        Returns:
            numpy.ndarray:
                values of the matching columns

        Raises:
            ValueError:
                if no columns match.
        """
        regex = r'{}(#[0-9]+)?$'.format(re.escape(column_name))
        columns = data.columns[data.columns.str.match(regex)]
        if columns.empty:
            raise ValueError('No columns match_ {}'.format(column_name))

        values = [data.pop(column).values for column in columns]

        if len(values) == 1:
            return values[0]

        return np.column_stack(values)

    def reverse_transform(self, data):
        """Revert the transformations back to the original values.

        Args:
            data (pandas.DataFrame):
                Data to revert.

        Returns:
            pandas.DataFrame:
                reversed data.
        """
        if self.copy:
            data = data.copy()

        for column_name, transformer in self._transformers.items():
            columns = self._get_columns(data, column_name)
            data[column_name] = transformer.reverse_transform(columns)

        return data
