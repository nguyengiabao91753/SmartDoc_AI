from app.database.sqlite_db import init_db
from app.core.logger import LOG

def start():
    init_db()
    LOG.info("SmartDoc AI core ready. Run Streamlit UI via `streamlit run ui/streamlit_app.py`")

if __name__ == "__main__":
    start()