# sql_server_pipeline

## setup .env file

in order to properly store the credentials to the database, you have to create a `.env` file with the following fields and your configuration:

```
SQL_SERVER_IP = "localhost"
SQL_SERVER_PORT = "1456"
DB = "metmast"
DB_USER = "metmast_user"
DB_PW = "password"
DB_DRIVER = "ODBC+Driver+18+for+SQL+Server"
```

## Database Setup with docker

It is possible to use your local database. Though, the easiest and fastest way to get started is, to setup a docker instance. Simply follow the steps below.

## initialize and run mssql docker from dockerhub
```
sudo docker run -e "ACCEPT_EULA=Y" -e "MSSQL_SA_PASSWORD=$password" -p 1456:1433 -d mcr.microsoft.com/mssql/server:2022-latest
```
Or look for an image of choice at https://hub.docker.com/_/microsoft-mssql-server

## connect to database from client and configure database
connect to db:
```
sqlcmd -S $ip_address,1456 -U sa -P "$password"
```

set most basic configuration:

```
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