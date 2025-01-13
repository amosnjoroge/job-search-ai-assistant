import streamlit as st
import sqlite3
import uuid
import warnings
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from sqlalchemy import text


class DataHandler:
    DB_NAME = "ajsa_database.sqlite3"
    TABLE_APPLICATIONS = "applications"
    TABLE_CURRENT_APPLICATION = "current_application"
    TABLE_ANALYSIS_STEPS = "analysis_steps"

    TABLES = {
        TABLE_APPLICATIONS: [
            "id TEXT PRIMARY KEY",
            "job_title TEXT",
            "company TEXT",
            "application_date TEXT",
            "update_date TEXT",
            "resume TEXT",
            "resume_regenerated TEXT",
            "cover_letter TEXT",
            "posting_url TEXT",
            "posting_text TEXT",
            "status TEXT",
            "CONSTRAINT id_unique UNIQUE (id)",
        ],
        TABLE_CURRENT_APPLICATION: [
            "id TEXT PRIMARY KEY",
            "application_id TEXT UNIQUE",
            "FOREIGN KEY(application_id) REFERENCES applications(id) ON DELETE CASCADE",
        ],
        TABLE_ANALYSIS_STEPS: [
            "id TEXT PRIMARY KEY",
            "application_id TEXT",
            "data TEXT",
            "created_at TEXT",
            "FOREIGN KEY(application_id) REFERENCES applications(id) ON DELETE CASCADE"
        ],
    }

    @classmethod
    def get_connection(cls):
        """Get SQLite connection using Streamlit's connection API"""
        return st.connection(
            "jobs_db",
            type="sql",
            url=f"sqlite:///{cls.DB_NAME}",
            autocommit=False,
        )

    @classmethod
    def initialize_database(cls):
        """Initialize database tables"""
        conn = cls.get_connection()

        try:
            with conn.session as s:
                # Enable WAL mode and other settings
                s.execute(text("PRAGMA journal_mode=WAL"))
                s.execute(text("PRAGMA busy_timeout=5000"))
                s.execute(text("PRAGMA foreign_keys = ON"))

                # Create all tables
                for table_name, columns in cls.TABLES.items():
                    columns_sql = ", ".join(columns)
                    query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql})"
                    s.execute(text(query))

                s.commit()
        except Exception as e:
            raise Exception(f"Database initialization failed: {str(e)}")

    @classmethod
    def reset_database(cls):
        """Reset database by dropping and recreating all tables"""
        conn = cls.get_connection()

        try:
            with conn.session as s:
                # Drop all tables
                for table_name in cls.TABLES:
                    s.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                s.commit()

            # Reinitialize database
            cls.initialize_database()

        except Exception as e:
            raise Exception(f"Database reset failed: {str(e)}")

    def query_data(
        self, table_name: str, condition: Optional[str] = None
    ) -> pd.DataFrame:
        """Query data from table with optional condition"""
        conn = self.get_connection()

        query = f"SELECT * FROM {table_name}"
        if condition:
            query += f" WHERE {condition}"

        return conn.query(query, ttl=0)  # ttl=0 means no caching

    def insert_data(
        self, table_name: str, data: tuple, columns: Optional[tuple[str]] = None
    ) -> str:
        """Insert data into table"""
        conn = self.get_connection()

        try:
            if columns is None:
                # Get columns from table info
                result = conn.query(f"PRAGMA table_info({table_name})")
                columns = tuple(row for row in result.name)

            # Generate UUID if needed
            if "id" not in columns:
                inserted_id = str(uuid.uuid4())
                columns = ("id",) + columns
                data = (inserted_id,) + data
            else:
                id_index = columns.index("id")
                inserted_id = data[id_index]

            # Create dictionary of named parameters
            data_dict = dict(zip(columns, data))

            # Prepare insert query with named parameters
            column_names = ", ".join(columns)
            placeholders = ", ".join([f":{col}" for col in columns])
            query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"

            with conn.session as s:
                s.execute(text(query), params=data_dict)
                s.commit()

            return inserted_id

        except Exception as e:
            raise e

    def update_data(self, table_name: str, data: Dict[str, Any], condition: str):
        """Update data in table"""
        conn = self.get_connection()

        try:
            # Validate columns
            result = conn.query(f"PRAGMA table_info({table_name})")
            valid_columns = {row for row in result.name}
            invalid_columns = set(data.keys()) - valid_columns
            if invalid_columns:
                raise ValueError(f"Invalid columns: {invalid_columns}")

            # Prepare update query with named parameters
            set_values = ", ".join([f"{key} = :{key}" for key in data.keys()])
            query = f"UPDATE {table_name} SET {set_values} WHERE {condition}"

            with conn.session as s:
                s.execute(text(query), params=data)
                s.commit()

        except Exception as e:
            raise e

    def delete_data(self, table_name: str, condition: str) -> bool:
        """Delete data from table"""
        conn = self.get_connection()

        try:
            query = f"DELETE FROM {table_name} WHERE {condition}"
            with conn.session as s:
                s.execute(text(query))
                s.commit()
            return True

        except Exception as e:
            raise e


# dh = DataHandler()
# DataHandler.reset_database()
# dh.insert_data("applications", ("test_date2222", "resume"), ("application_date", "resume"))
# dh.update_data(
#     "applications",
#     {"job_title": "my jsdhaskj"},
#     "id = '4541c0d9-d890-42d4-b0ac-786fb5c1f7c5'",
# )

# dh.delete_data("applications", "id = '911f7efe-a3e5-4a4d-a2a3-e87e203d1076'")
