import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import configparser

# Read the password from config.properties
config = configparser.ConfigParser()
config.read('config.properties')
password = quote_plus(config.get('DEFAULT', 'db.password'))

# Connection string for MySQL on localhost, port 3306
connection_string = f"mysql+pymysql://root:{password}@localhost:3306/set_local"
engine = create_engine(connection_string)

