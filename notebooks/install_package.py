import sqlmlutils
connection = sqlmlutils.ConnectionInfo(server="localhost", database="metmast_0_4")#, uid="username", pwd="password"))
sqlmlutils.SQLPackageManager(connection).install("pandas")