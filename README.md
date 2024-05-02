# sql_server_pipeline

## Setup .env file

In order to properly store the credentials to the database, you have to create a `.env` file with the following fields and your configuration:

```conf
SQL_SERVER_IP = "localhost"
SQL_SERVER_PORT = "1456"
DB = "metmast"
DB_USER = "metmast_user"
DB_PW = "password"
DB_DRIVER = "ODBC+Driver+18+for+SQL+Server"
```

on linux the driver will be `/opt/microsoft/msodbcsql18/lib64/libmsodbcsql-18.3.so.2.1` or equivalent.
alternatively you can search for it with `find / -type f -iname "libmsodbcsql*" 2>/dev/null`

## Database setup with docker

It is possible to use your local database. Though, the easiest and fastest way to get started is to setup a docker instance. Simply follow the steps below.

## Initialize and run mssql docker from dockerhub

```sh
sudo docker run -e "ACCEPT_EULA=Y" -e "MSSQL_SA_PASSWORD=$password" -p 1456:1433 -d mcr.microsoft.com/mssql/server:2022-latest
```

or look for an image of choice at <https://hub.docker.com/_/microsoft-mssql-server>

## Connect to database from client and configure database

connect to db:

```sh
sqlcmd -S $ip_address,1456 -C -U sa -P "$password"
```

set most basic configuration:

```sql
CREATE DATABASE metmast;
GO
CREATE LOGIN metmast_user WITH PASSWORD = '$db_password';
GO
USE metmast;
CREATE USER metmast_user FOR LOGIN metmast_user;
GO
ALTER ROLE db_owner ADD MEMBER metmast_user;
GO
```

## Extracting data

```sh
unzip data.zip
```

## Installing dependencies

```sh
# optional: create and use a venv
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Order to run the notebooks in

1. `SQL_Setp_custom1`
2. `CreateTablesData`
3. `LoadDataSet`
4. `CreateTablesTrain`
5. ...
