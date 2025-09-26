
import streamlit as st
import os
import glob
import psycopg2

conn = psycopg2.connect(
    dbname="postgres",
    user="admin",
    password="admin",
    host="localhost",
    port="5432"
)

# Рекурсивно ищем все файлы в data/ и подпапках
documents = [os.path.relpath(f, "data/") for f in glob.glob("data/**/*", recursive=True) if os.path.isfile(f)]

st.title("Удаление документов")

selected_docs = st.multiselect("Выберите документы для удаления:", documents)

if st.button("Удалить выбранные"):
    for doc in selected_docs:
        try:
            os.remove(os.path.join("data/", doc))
            st.success(f"Документ {doc} удалён.")
        except Exception as e:
            st.error(f"Ошибка при удалении {doc}: {e}")

import streamlit as st

st.title("Удаление таблиц из базы данных")

# Получаем список таблиц
with conn.cursor() as cur:
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    tables = [row[0] for row in cur.fetchall()]

selected_tables = st.multiselect("Выберите таблицы для удаления:", tables)

if st.button("Удалить выбранные таблицы"):
    with conn.cursor() as cur:
        for table in selected_tables:
            try:
                cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE;')
                st.success(f"Таблица {table} удалена.")
            except Exception as e:
                st.error(f"Ошибка при удалении {table}: {e}")
        conn.commit()

conn.close()