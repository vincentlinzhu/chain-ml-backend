import mysql.connector

# returns a MySQL database connection object
def connect_to_db():
    """Connect to the SingleStore database."""
    return mysql.connector.connect(
        host='your_host',       # e.g., 'localhost' or the IP address of your SingleStore instance
        user='your_user',       # e.g., 'root' or your SingleStore username
        password='your_password',  # Your password
        database='your_database'   # The database you are working with
    )