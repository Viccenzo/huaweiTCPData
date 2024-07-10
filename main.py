from pyModbusTCP.client import ModbusClient
import numpy as np
import pandas as pd
import sqlalchemy
import numpy as np
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    DateTime,
    exists,
    inspect,
    text,
    Integer, 
    String, 
    Float, 
    Boolean
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import VARCHAR
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.dialects.postgresql import insert
import time

def createEngine():    
    engine = create_engine(
        "postgresql://fotovoltaica:fotovoltaica123@150.162.142.84/fotovotlaica", echo=False
    )
    return engine

huawei65 = ModbusClient(host="150.162.142.65", port=502, unit_id=0, auto_open=True)
huawei64 = ModbusClient(host="150.162.142.64", port=502, unit_id=0, auto_open=True)
huawei63 = ModbusClient(host="150.162.142.63", port=502, unit_id=0, auto_open=True)

data = {}

# Function to map types from pandas to SQLalchey
def map_dtype(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return Integer
    elif pd.api.types.is_float_dtype(dtype):
        return Float
    elif pd.api.types.is_bool_dtype(dtype):
        return Boolean
    else:
        return VARCHAR

def insert_dataframe(engine, table_name, dataframe):

    # Convert dataframe to dictionary format
    records = dataframe.to_dict(orient='records')
    
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)

    with engine.connect() as conn:
        with conn.begin():
            for record in records:
                stmt = insert(table).values(record)
                stmt = stmt.on_conflict_do_nothing(index_elements=['timestamp'])  # Adjust this for your primary key column(s)
                conn.execute(stmt)

def uploadToDB(engine, dataframe, table_name):
    # Ensure the TIMESTAMP column exists and is set as primary key (check if it is still necessary)
    """with engine.connect() as connection:
        # Using a transaction to ensure atomicity
        with connection.begin():
            # Check if the column TIMESTAMP exists and add primary key if not present
            inspector = inspect(engine)
            pk_info = inspector.get_pk_constraint(table_name)
            if 'TIMESTAMP' not in pk_info['constrained_columns']:
                try:
                    alter_pk_sql = text(
                        f'ALTER TABLE "{table_name}" ADD PRIMARY KEY ("TIMESTAMP");'
                    )
                    connection.execute(alter_pk_sql)
                    print("Column TIMESTAMP set as primary key.")
                except ProgrammingError as e:
                    print(f"Error setting TIMESTAMP as primary key: {e}")

            # Update column type to TIMESTAMP
            alter_sql = text(
                f'ALTER TABLE "{table_name}" ALTER COLUMN "TIMESTAMP" TYPE TIMESTAMP USING "TIMESTAMP"::timestamp;'
            )
            connection.execute(alter_sql)
            print("Column TIMESTAMP updated to datetime type.") """

    try:    #Try to incert data all at once (faster)
        dataframe.to_sql(table_name, engine, if_exists='append', index=False)
        print(f"Data successfully uploaded to a newly created {table_name} table!")
    except: # Insert data with conflict handling (Slow)
        print(f'primary key conflict, atempeting to insert data avoiding conflicts')
        records = dataframe.to_dict(orient='records')

        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=engine)

        with engine.connect() as conn:
            with conn.begin():
                for record in records:
                    stmt = insert(table).values(record)
                    stmt = stmt.on_conflict_do_nothing(index_elements=['timestamp'])
                    conn.execute(stmt)

# Function to check if a table exists inside the database
def tableExists(tableName, engine):
    ins = inspect(engine)
    ret =ins.dialect.has_table(engine.connect(),tableName)
    #print('Table "{}" exists: {}'.format(tableName, ret))
    return ret

# function to create a table on database
def createTable(dataFrame, engine, tableName):
    column_types = {name: map_dtype(dtype) for name, dtype in dataFrame.dtypes.items()}
    metadata = MetaData()
    table = Table(tableName, metadata, *(Column(name, column_types[name]) for name in column_types))
    metadata.create_all(engine)

# function that compare header between dataframe and databse to extract diferences
def headerMismach(tableName, engine, dataFrame):
    inspector = inspect(engine)
    columns = inspector.get_columns(tableName)
    column_names = [column['name'] for column in columns]
    headers = dataFrame.columns.tolist()
    missing_in_db = set(headers) - set(column_names)
    return missing_in_db

# function that adds missing columns present on pandas dataframe
def addMissingColumn(missing_columns,engine,table_name,dataFrame):
    metadata = MetaData()
    for column_name in missing_columns:
        column_types = {name: map_dtype(dtype) for name, dtype in dataFrame.dtypes.items()}
        column_type = column_types[column_name]
        alter_statement = text(f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {column_type().compile(dialect=engine.dialect)}')
        try:
            with engine.connect() as conn:
                with conn.begin():
                    conn.execute(alter_statement)
            print(f"column '{column_name}' sucessfully added")
        except ProgrammingError as e:
            print(f"Column creation error '{column_name}': {e}")

def primaryKeyExists(engine, tableName):
    metadata = MetaData()
    inspector = inspect(engine)
    primary_keys = inspector.get_pk_constraint(tableName)['constrained_columns']
    return primary_keys

def addPrimarykey(engine, tableName, keyName):
    alter_statement = text(f'ALTER TABLE "{tableName}" ADD PRIMARY KEY ("{keyName}")')
    try:
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(alter_statement)
    except ProgrammingError as e:
        print(f"Set primary key error: {e}")

def readHuawei(equipment, varName):
    global data
    data[varName] = {}
    # Time UTC
    regs = equipment.read_holding_registers(40000,2)
    data[varName]["timestamp"] = np.uint32(np.uint32(regs[0]<<16)+np.uint32(regs[1]))

    # input power in kW
    regs = equipment.read_holding_registers(40521,2)
    data[varName]["inputPower"] = np.uint32(np.uint32(regs[0]<<16)+np.uint32(regs[1]))

    # Active power in kW
    regs = equipment.read_holding_registers(40525,2)
    data[varName]["activePower"] = np.int32(np.uint32(regs[0]<<16)+np.uint32(regs[1]))

    # Power factor in kW
    regs = equipment.read_holding_registers(40532,1)
    data[varName]["powerFactor"] = np.int32(regs[0])

    # CO2 reduction in kg
    regs = equipment.read_holding_registers(40550,4)
    data[varName]["cO2Reduction"] = np.uint32(np.uint64(regs[0]<<48)+np.uint64(regs[1]<<32)+np.uint64(regs[2]<<16)+np.uint64(regs[3]))

    # dCCurrent n A
    regs = equipment.read_holding_registers(40554,2)
    data[varName]["activePower"] = np.int32(np.uint32(regs[0]<<16)+np.uint32(regs[1]))

    # eTotal in kWh
    regs = equipment.read_holding_registers(40560,2)
    data[varName]["eTotal"] = np.uint32(np.uint32(regs[0]<<16)+np.uint32(regs[1]))

    # eDaily in kWh
    regs = equipment.read_holding_registers(40562,2)
    data[varName]["eDaily"] = np.uint32(np.uint32(regs[0]<<16)+np.uint32(regs[1]))

    # dailyPowerGeneration in h
    regs = equipment.read_holding_registers(40564,2)
    data[varName]["dailyPowerGeneration"] = np.uint32(np.uint32(regs[0]<<16)+np.uint32(regs[1]))

    # phaseACurrent in A
    regs = equipment.read_holding_registers(40572,1)
    data[varName]["phaseACurrent"] = np.int32(regs[0])

    # phaseBCurrent in A
    regs = equipment.read_holding_registers(40573,1)
    data[varName]["phaseBCurrent"] = np.int32(regs[0])

    # phaseCCurrent in A
    regs = equipment.read_holding_registers(40574,1)
    data[varName]["phaseCurrent"] = np.int32(regs[0])

    # Uab in V
    regs = equipment.read_holding_registers(40575,1)
    data[varName]["Uab"] = np.uint32(regs[0])

    # Uabc in V
    regs = equipment.read_holding_registers(40576,1)
    data[varName]["Ubc"] = np.uint32(regs[0])

    # Uca in V
    regs = equipment.read_holding_registers(40577,1)
    data[varName]["Uca"] = np.uint32(regs[0])

    # inverterEfficiency in %
    regs = equipment.read_holding_registers(40685,1)
    data[varName]["inverterEfficiency"] = np.uint32(regs[0])

    # dIStatus
    regs = equipment.read_holding_registers(40700,1)
    data[varName]["dIStatus"] = np.uint32(regs[0])

    # alarmInfo1
    regs = equipment.read_holding_registers(50000,1)
    data[varName]["alarmInfo1"] = np.uint32(regs[0])

    # alarmInfo2
    regs = equipment.read_holding_registers(50000,1)
    data[varName]["alarmInfo1"] = np.uint32(regs[0])


def main():
    global data

    engine = createEngine()

    while 1:
        readHuawei(huawei65,"huawei65")
        readHuawei(huawei64,"huawei64")
        readHuawei(huawei63,"huawei63")
        for equipment in data:
            print(data[equipment])
            fileName = equipment
            df = pd.DataFrame(data[equipment], index=[0])

            if not tableExists(fileName,engine):
                print('The table does not exist, creating table')
                createTable(df, engine, fileName)

            missmach = headerMismach(fileName,engine,df)
            if(len(missmach) !=0):
                print(f'{len(missmach)} mismach were found, addin headers to database')
                addMissingColumn(missmach,engine,fileName,df)
            
            primaryKey = primaryKeyExists(engine,fileName)
            if primaryKey == []:
                print(primaryKey)
                addPrimarykey(engine,fileName,'timestamp') # for now the only primary key is going to be timestamp, changein the future
            
            print("uploading data to database")
            uploadToDB(engine,df,fileName)
        time.sleep(60)


main()