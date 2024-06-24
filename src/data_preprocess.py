from typing import List
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import DataType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import isnan, when, count
from pyspark.sql.types import FloatType, IntegerType, ByteType


features_cols = [
    "type_indx",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "newbalanceDest",
    "oldbalanceDest",
]

target_col = "isFraud"


def cast_df_values(df: DataFrame, columns: List[str], newType: DataType) -> DataFrame:
    for col in columns:
        df = df.withColumn(col, df[col].cast(newType()))

    return df


def encode_string_column(df: DataFrame, inputCol: str, outputCol: str):
    indexer = StringIndexer(inputCol=inputCol, outputCol=outputCol)
    pipeline = Pipeline(stages=[indexer])

    model = pipeline.fit(df)
    df = model.transform(df)
    return df


def get_data_shape(df: DataFrame):
    return df.count(), len(df.columns)


def count_nans(df: DataFrame):
    return df.select([count(when(isnan(c), c)).alias(c) for c in df.columns])


def count_col_values(df: DataFrame, col: str):
    return df.groupby(col).count()


def assemble_cols(df: DataFrame, inputCols, outputCol):
    assembler = VectorAssembler(inputCols=inputCols, outputCol=outputCol)
    return assembler.transform(df)


def preprocess_data(df: DataFrame):
    float_columns = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    df = cast_df_values(df, float_columns, newType=FloatType)

    int_columns = ["step", "isFraud", "isFlaggedFraud"]
    df = cast_df_values(df, int_columns, newType=IntegerType)

    byte_columns = ["nameOrig", "nameDest"]
    df = cast_df_values(df, byte_columns, newType=ByteType)

    df = encode_string_column(df, "type", "type_indx")

    df = assemble_cols(df, features_cols, "features")

    return df
