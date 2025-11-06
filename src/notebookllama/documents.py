from dataclasses import dataclass
from sqlalchemy import (
    Table,
    MetaData,
    Column,
    Text,
    Integer,
    create_engine,
    Engine,
    Connection,
    insert,
    select,
    text,
)
from typing import Optional, List, cast, Union

# Set PostgreSQL client encoding environment variable BEFORE importing psycopg2
# This ensures psycopg2 uses UTF-8 encoding from the start
import os
os.environ['PGCLIENTENCODING'] = 'UTF8'


def apply_string_correction(string: str) -> str:
    return string.replace("''", "'").replace('""', '"')


@dataclass
class ManagedDocument:
    document_name: str
    content: str
    summary: str
    q_and_a: str
    mindmap: str
    bullet_points: str


class DocumentManager:
    def __init__(
        self,
        engine: Optional[Engine] = None,
        engine_url: Optional[str] = None,
        table_name: Optional[str] = None,
        table_metadata: Optional[MetaData] = None,
    ):
        self.table_name: str = table_name or "documents"
        self._table: Optional[Table] = None
        self._connection: Optional[Connection] = None
        self.metadata: MetaData = cast(MetaData, table_metadata or MetaData())
        # Allow running without a configured DB. If engine_url contains 'None' or is falsy,
        # treat DB as not configured and make put_documents a no-op with informative logging.
        if engine:
            self._engine: Union[Engine, str, None] = engine
        elif engine_url and isinstance(engine_url, str) and "None" not in engine_url:
            self._engine = engine_url
        else:
            # No DB configured; we'll skip DB writes but keep manager usable for reads (returns empty)
            self._engine = None

    @property
    def connection(self) -> Connection:
        if not self._connection:
            self._connect()
        return cast(Connection, self._connection)

    @property
    def table(self) -> Table:
        if self._table is None:
            self._create_table()
        return cast(Table, self._table)

    def _connect(self) -> None:
        # move network calls outside of constructor
        if not self._engine:
            # DB not configured; nothing to connect
            return
        
        # Ensure PGCLIENTENCODING is set (should be set at module level, but double-check)
        import os as os_module
        os_module.environ['PGCLIENTENCODING'] = 'UTF8'
        
        if isinstance(self._engine, str):
            # Add connection arguments to ensure UTF-8 encoding
            # For psycopg2, we need to set encoding explicitly to avoid decode errors
            # Add client_encoding to connection string if not already present
            engine_url = self._engine
            if 'psycopg2' in engine_url and 'client_encoding' not in engine_url:
                # Add client_encoding parameter to the connection string
                separator = '&' if '?' in engine_url else '?'
                engine_url = f"{engine_url}{separator}client_encoding=UTF8"
            
            # Create engine with explicit encoding settings
            try:
                self._engine = create_engine(
                    engine_url,
                    connect_args={
                        'client_encoding': 'UTF8',
                        'options': '-c client_encoding=UTF8'
                    } if 'psycopg2' in engine_url else {},
                    pool_pre_ping=True,  # Verify connections before using
                    # Set encoding at connection pool level
                    poolclass=None,  # Use default pool
                )
            except Exception as e:
                # If engine creation fails, try with minimal settings
                import logging
                logging.warning(f"Engine creation failed: {e}. Retrying with minimal settings...")
                self._engine = create_engine(engine_url, pool_pre_ping=True)
        
        try:
            self._connection = self._engine.connect()
            # Ensure the DB client encoding is UTF8 to avoid codec mismatch when DB returns bytes
            try:
                # Works for Postgres/psycopg2 - set encoding explicitly
                self._connection.execute(text("SET client_encoding = 'UTF8';"))
                # Also set server encoding if possible
                try:
                    self._connection.execute(text("SHOW server_encoding;"))
                except Exception:
                    pass
            except Exception:
                # best-effort; ignore if not supported by the backend
                pass
        except (UnicodeDecodeError, UnicodeError) as e:
            # If we get a decode error during connection, try to reconnect with explicit encoding
            import logging
            logging.warning(f"Unicode decode error during connection: {e}. Retrying with explicit encoding...")
            # Close any partial connection
            try:
                if hasattr(self, '_connection') and self._connection:
                    self._connection.close()
            except Exception:
                pass
            
            # Try reconnecting with forced UTF-8
            if isinstance(self._engine, str):
                # Force UTF-8 in connection string
                base_url = self._engine.split('?')[0].split('&')[0]
                engine_url = f"{base_url}?client_encoding=UTF8"
                self._engine = create_engine(
                    engine_url,
                    connect_args={'client_encoding': 'UTF8'},
                    pool_pre_ping=True,
                )
            self._connection = self._engine.connect()
        except Exception as e:
            # Catch any other connection errors
            import logging
            logging.error(f"Connection error: {e}")
            raise

    def _create_table(self) -> None:
        # Define the table in metadata and create via metadata.create_all for better backend support
        self._table = Table(
            self.table_name,
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("document_name", Text),
            Column("content", Text),
            Column("summary", Text),
            Column("q_and_a", Text),
            Column("mindmap", Text),
            Column("bullet_points", Text),
        )
        # Use metadata.create_all which is backend-agnostic and respects checkfirst
        try:
            self.metadata.create_all(self._engine, checkfirst=True)
        except Exception:
            # Fallback to creating via connection if engine-based creation fails
            if self.connection is not None:
                self._table.create(self.connection, checkfirst=True)

    def put_documents(self, documents: List[ManagedDocument]) -> None:
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("DocumentManager")
        if not self._engine:
            logger.info("No DB engine configured — skipping put_documents (in-memory only).")
            return
        # Prepare rows and perform a batch insert inside a transaction using engine.begin()
        rows = []

        def safe_str(val):
            # Normalize any value to a UTF-8-valid Python str. Replace invalid bytes.
            try:
                if val is None:
                    return ""
                if isinstance(val, bytes):
                    # decode bytes, try utf-8 then fallback to latin-1, finally replace
                    try:
                        s = val.decode('utf-8')
                    except Exception:
                        try:
                            s = val.decode('latin-1')
                        except Exception:
                            s = val.decode('utf-8', errors='replace')
                else:
                    s = str(val)
                # Re-encode/decode to ensure any surrogate or invalid chars are replaced
                return s.encode('utf-8', errors='replace').decode('utf-8')
            except Exception:
                return ""

        for document in documents:
            doc_name = safe_str(document.document_name)
            content = safe_str(document.content)
            summary = safe_str(document.summary)
            q_and_a = safe_str(document.q_and_a)
            mindmap = safe_str(document.mindmap)
            bullet_points = safe_str(document.bullet_points)

            logger.info(f"Запись документа в БД: {doc_name} (content_len={len(content)}, summary_len={len(summary)})")
            rows.append(
                {
                    "document_name": doc_name,
                    "content": content,
                    "summary": summary,
                    "q_and_a": q_and_a,
                    "mindmap": mindmap,
                    "bullet_points": bullet_points,
                }
            )

        if not rows:
            logger.info("Нет документов для записи.")
            return

        # Use engine-level transaction to insert all rows. This avoids explicit connection.commit()
        try:
            # Ensure engine is available. If _engine was a URL string, create an Engine and persist it.
            if isinstance(self._engine, str):
                engine = create_engine(self._engine)
                # persist created engine for future calls
                self._engine = engine
            else:
                engine = self._engine

            with engine.begin() as conn:
                # Ensure the DB client encoding is UTF8 on this transactional connection
                try:
                    # Use exec_driver_sql to send raw SQL to the DB driver
                    conn.exec_driver_sql("SET client_encoding = 'UTF8';")
                except Exception:
                    # best-effort; ignore if not supported
                    pass
                # Defensive: ensure all values are plain str to avoid DB driver decode issues
                sanitized_rows = []
                for r in rows:
                    san = {}
                    for k, v in r.items():
                        if not isinstance(v, str):
                            try:
                                v = str(v)
                            except Exception:
                                v = ""
                        san[k] = v
                    sanitized_rows.append(san)
                try:
                    conn.execute(insert(self.table), sanitized_rows)
                except Exception as ie:
                    import traceback

                    traceback.print_exc()
                    # Log a helpful sample of the problematic data
                    try:
                        sample = sanitized_rows[0]
                    except Exception:
                        sample = sanitized_rows
                    logger.error("Batch insert failed. Sample row (repr): %s", repr(sample))
                    raise
            logger.info("Коммит транзакции завершён.")
        except Exception as e:
            logger.error(f"Ошибка выполнения batch insert/commit: {e}")
            raise

    def get_documents(self, names: Optional[List[str]] = None) -> List[ManagedDocument]:
        if self.table is None:
            self._create_table()
        if not names:
            stmt = select(self.table).order_by(self.table.c.id)
        else:
            stmt = (
                select(self.table)
                .where(self.table.c.document_name.in_(names))
                .order_by(self.table.c.id)
            )
        result = self.connection.execute(stmt)
        rows = result.fetchall()
        documents = []
        
        def safe_str(val):
            # Normalize any value to a UTF-8-valid Python str. Replace invalid bytes.
            try:
                if val is None:
                    return ""
                if isinstance(val, bytes):
                    # decode bytes, try utf-8 then fallback to latin-1, finally replace
                    try:
                        s = val.decode('utf-8')
                    except Exception:
                        try:
                            s = val.decode('latin-1')
                        except Exception:
                            s = val.decode('utf-8', errors='replace')
                else:
                    s = str(val)
                # Re-encode/decode to ensure any surrogate or invalid chars are replaced
                return s.encode('utf-8', errors='replace').decode('utf-8')
            except Exception:
                return ""
        
        for row in rows:
            documents.append(
                ManagedDocument(
                    document_name=safe_str(row.document_name),
                    content=safe_str(row.content),
                    summary=safe_str(row.summary),
                    q_and_a=safe_str(row.q_and_a),
                    mindmap=safe_str(row.mindmap),
                    bullet_points=safe_str(row.bullet_points),
                )
            )
        return documents

    def get_names(self) -> List[str]:
        if self.table is None:
            self._create_table()
        stmt = select(self.table)
        result = self.connection.execute(stmt)
        rows = result.fetchall()
        
        def safe_str(val):
            # Normalize any value to a UTF-8-valid Python str. Replace invalid bytes.
            try:
                if val is None:
                    return ""
                if isinstance(val, bytes):
                    # decode bytes, try utf-8 then fallback to latin-1, finally replace
                    try:
                        s = val.decode('utf-8')
                    except Exception:
                        try:
                            s = val.decode('latin-1')
                        except Exception:
                            s = val.decode('utf-8', errors='replace')
                else:
                    s = str(val)
                # Re-encode/decode to ensure any surrogate or invalid chars are replaced
                return s.encode('utf-8', errors='replace').decode('utf-8')
            except Exception:
                return ""
        
        return [safe_str(row.document_name) for row in rows]

    def disconnect(self) -> None:
        if not self._connection:
            raise ValueError("Engine was never connected!")
        if isinstance(self._engine, str):
            pass
        else:
            self._engine.dispose(close=True)
