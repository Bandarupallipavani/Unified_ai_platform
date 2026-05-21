"""
ai/nl_connector.py — Natural Language Data Connector
Interprets plain-English descriptions to connect databases, REST APIs, files, cloud storage.
"""
import os
import json
import logging
from typing import Optional

import anthropic
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


async def interpret_connector(description: str, session_id: str) -> dict:
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[{
            "role": "user",
            "content": f"""
Parse this data source description and return a JSON object.

Description: "{description}"

Return ONLY a JSON object with:
- "connector_type": one of ["postgresql","mysql","sqlite","bigquery","rest_api","csv_url","s3","demo"]
- "connection_params": relevant params (host, port, database, table, url, etc.)
- "sql_query": if applicable
- "explanation": one sentence describing what will happen

If ambiguous, use "demo" connector. Return only valid JSON.
"""
        }]
    )

    text = response.content[0].text.strip()
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {"connector_type": "demo", "explanation": text[:200]}

    connector_type = parsed.get("connector_type", "demo")
    params = parsed.get("connection_params", {})
    query = parsed.get("sql_query")

    df = await _execute_connector(connector_type, params, query, description)

    return {
        "dataframe": df,
        "query": query,
        "connector_type": connector_type,
        "explanation": parsed.get("explanation", ""),
    }


async def _execute_connector(connector_type, params, query, description) -> pd.DataFrame:
    if connector_type == "postgresql":
        return _connect_postgresql(params, query)
    elif connector_type == "mysql":
        return _connect_mysql(params, query)
    elif connector_type == "sqlite":
        return _connect_sqlite(params, query)
    elif connector_type == "rest_api":
        return await _connect_rest_api(params)
    elif connector_type == "csv_url":
        return _connect_csv_url(params.get("url", ""))
    elif connector_type == "bigquery":
        return _connect_bigquery(params, query)
    else:
        return _generate_demo_data(description)


def _connect_postgresql(params, query):
    import sqlalchemy
    conn_str = f"postgresql://{params.get('user','')}:{params.get('password','')}@{params.get('host','localhost')}:{params.get('port',5432)}/{params.get('database','')}"
    engine = sqlalchemy.create_engine(conn_str)
    return pd.read_sql(query or f"SELECT * FROM {params.get('table','main')} LIMIT 10000", engine)


def _connect_mysql(params, query):
    import sqlalchemy
    conn_str = f"mysql+pymysql://{params.get('user','')}:{params.get('password','')}@{params.get('host','localhost')}:{params.get('port',3306)}/{params.get('database','')}"
    engine = sqlalchemy.create_engine(conn_str)
    return pd.read_sql(query or f"SELECT * FROM {params.get('table','main')} LIMIT 10000", engine)


def _connect_sqlite(params, query):
    import sqlite3
    conn = sqlite3.connect(params.get("path", params.get("database", "")))
    return pd.read_sql(query or f"SELECT * FROM {params.get('table','main')} LIMIT 10000", conn)


async def _connect_rest_api(params) -> pd.DataFrame:
    import httpx
    async with httpx.AsyncClient() as c:
        resp = await c.get(params.get("url", ""), headers=params.get("headers", {}), timeout=30)
        data = resp.json()
    if isinstance(data, list):
        return pd.DataFrame(data)
    for key in ["data", "results", "items", "records"]:
        if key in data and isinstance(data[key], list):
            return pd.DataFrame(data[key])
    return pd.DataFrame([data])


def _connect_csv_url(url):
    return pd.read_csv(url)


def _connect_bigquery(params, query):
    from google.cloud import bigquery
    client_bq = bigquery.Client(project=params.get("project_id", ""))
    q = query or f"SELECT * FROM `{params.get('project_id','')}.{params.get('dataset','')}.{params.get('table','')}` LIMIT 10000"
    return client_bq.query(q).to_dataframe()


def _generate_demo_data(description: str) -> pd.DataFrame:
    n = 500
    df = pd.DataFrame({
        "id": range(1, n + 1),
        "feature_1": np.random.normal(50, 15, n),
        "feature_2": np.random.exponential(10, n),
        "feature_3": np.random.uniform(0, 100, n),
        "category": np.random.choice(["A", "B", "C", "D"], n),
        "text_field": [f"Sample text record {i} for demo" for i in range(n)],
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    logger.info(f"Generated demo dataset ({n} rows) for: {description[:60]}")
    return df
