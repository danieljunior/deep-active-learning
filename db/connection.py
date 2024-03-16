from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker
from base import Base
from document import Document

def get_engine():
    url = URL.create(
        drivername="postgresql",
        username="provenance",
        password="Provenance",
        host="db",
        database="al_provenance"
    )

    return create_engine(url)

def get_session(engine=None):
    if not engine:
        engine = create_engine()
    Session = sessionmaker(bind=engine)
    return Session()
# connection = engine.connect()

def main():
    engine = get_engine()
    Base.metadata.create_all(engine)
    session = get_session(engine)
    print(session.query(Document).all())

if __name__ == '__main__':
    main()