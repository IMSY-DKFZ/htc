# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

# Copyright (c) 2013 Yhat, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import functools
import hashlib
import inspect
import io
import re
import sqlite3
from contextlib import contextmanager
from typing import Callable
from warnings import catch_warnings, filterwarnings

import numpy as np
from pandas.io.sql import read_sql, to_sql

# Based on: https://github.com/yhat/pandasql/


# Allow numpy arrays and lists to be passed to sqldf https://github.com/pandas-dev/pandas/issues/29240
def adapt_array(arr):
    # http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    arr = np.asarray(arr)
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_adapter(list, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("arraytodb", convert_array)


class NumpyFunctions:
    def __init__(self, numpy_func: Callable):
        self.values = []
        self.numpy_func = numpy_func
        self.args = []

    def step(self, value, *args):
        if value is None:
            return

        self.values.append(value)
        self.args = args

    def finalize(self):
        kwargs = {}
        for arg in self.args:
            name, value = re.split(r"\s*=\s*", arg)
            try:
                value = float(value)
            except ValueError:
                pass

            kwargs[name] = value

        return self.numpy_func(self.values, **kwargs)


def numpy_sql_class(numpy_func: Callable):
    class NumpyFunction(NumpyFunctions):
        __init__ = functools.partialmethod(NumpyFunctions.__init__, numpy_func=numpy_func)

    return NumpyFunction


__all__ = ["PandaSQL", "sqldf"]


class PandaSQL:
    def __init__(self, persist=False):
        """
        Initialize with a specific database.

        Args:
            persist: keep tables in database between different calls on the same object of this class.
        """
        self.persist = persist
        self.loaded_tables = set()
        if self.persist:
            self._conn = sqlite3.connect(":memory:")

    def __call__(self, query, env=None):
        """
        Execute the SQL query. Automatically creates tables mentioned in the query from dataframes before executing.

        Args:
            query: SQL query string, which can reference pandas dataframes as SQL tables.
            env: Variables environment - a dict mapping table names to pandas dataframes. If not specified use local and global variables of the caller.

        Returns: Pandas dataframe with the result of the SQL query.
        """
        if env is None:
            env = get_outer_frame_variables()

        with self.conn as conn:
            self._add_extensions(conn)

            for table_name in extract_table_names(query):
                if table_name not in env:
                    # don't raise error because the table may be already in the database
                    continue
                if self.persist and table_name in self.loaded_tables:
                    # table was loaded before using the same instance, don't do it again
                    continue
                self.loaded_tables.add(table_name)
                df = env[table_name]
                if "*" not in query.replace("(*)", ""):  # COUNT(*) is ok but * not
                    # Remove columns which are not used in the query (faster and type safer)
                    used_columns = [c for c in df.columns if c in query]
                    df = df[used_columns]
                write_table(df, table_name, conn)

            result = read_sql(query, conn)

        return result

    @property
    @contextmanager
    def conn(self):
        if self.persist:
            # the connection is created in __init__, so just return it
            yield self._conn
            # no cleanup needed
        else:
            # create the connection
            conn = sqlite3.connect(":memory:")
            conn.text_factory = str
            try:
                yield conn
            finally:
                # cleanup - close connection on exit
                conn.close()

    def _add_extensions(self, conn):
        mapping = {
            "std": np.std,
            "median": np.median,
            "quantile": np.quantile,
        }

        for name, func in mapping.items():
            conn.create_aggregate(name, -1, numpy_sql_class(func))

        conn.create_function("sha1", 1, lambda x: hashlib.sha1(bytes(x, "utf-8")).hexdigest())


def get_outer_frame_variables():
    """Get a dict of local and global variables of the first outer frame from another file."""
    cur_filename = inspect.getframeinfo(inspect.currentframe()).filename
    outer_frame = next(f for f in inspect.getouterframes(inspect.currentframe()) if f.filename != cur_filename)
    variables = {}
    variables.update(outer_frame.frame.f_globals)
    variables.update(outer_frame.frame.f_locals)
    return variables


def extract_table_names(query):
    """Extract table names from an SQL query."""
    # a good old fashioned regex. turns out this worked better than actually parsing the code
    tables_blocks = re.findall(r"(?:FROM|JOIN)\s+(\w+(?:\s*,\s*\w+)*)", query, re.IGNORECASE)
    tables = [tbl for block in tables_blocks for tbl in re.findall(r"\w+", block)]
    return set(tables)


def write_table(df, tablename, conn):
    """Write a dataframe to the database."""
    with catch_warnings():
        filterwarnings(
            "ignore", message="The provided table name '%s' is not found exactly as such in the database" % tablename
        )
        to_sql(
            df, name=tablename, con=conn, index=not any(name is None for name in df.index.names)
        )  # load index into db if all levels are named


def sqldf(query, env=None):
    """
    Query pandas data frames using sql syntax This function is meant for backward compatibility only. New users are encouraged to use the PandaSQL class.

    Parameters
    ----------
    query: string
        a sql query using DataFrames as tables
    env: locals() or globals()
        variable environment; locals() or globals() in your function
        allows sqldf to access the variables in your python environment

    Returns
    -------
    result: DataFrame
        returns a DataFrame with your query's result

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "x": range(100),
    ...     "y": range(100)
    ... })
    >>> from htc.utils.sqldf import sqldf
    >>> sqldf("SELECT AVG(x) FROM df").loc[0].item()
    49.5
    """
    with catch_warnings():
        # TODO: should be fixed in a general rewrite of this class
        filterwarnings("ignore", message="pandas only support SQLAlchemy connectable", category=UserWarning)

        return PandaSQL()(query, env)
