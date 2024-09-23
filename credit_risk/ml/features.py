from pyspark.sql import DataFrame,Window
from pyspark.sql.functions import col, udf,when
from pyspark.sql.types import StringType,IntegerType
from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from typing import List
from pyspark import keyword_only
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler

class DataCleaner(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, age_column: str = None, age_threshold: int = None, column_to_drop: str = None, emp_length_column: str = None, emp_length_threshold: int = None):
        super(DataCleaner, self).__init__()
        kwargs = self._input_kwargs
        self.age_column = Param(self, "age_column", "Column for person's age")
        self.age_threshold = Param(self, "age_threshold", "Age threshold to filter data")
        self.column_to_drop = Param(self, "column_to_drop", "Column to drop and reset index")
        self.emp_length_column = Param(self, "emp_length_column", "Column for person's employment length")
        self.emp_length_threshold = Param(self, "emp_length_threshold", "Employment length threshold to filter data")
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, age_column: str = None, age_threshold: int = None, column_to_drop: str = None, emp_length_column: str = None, emp_length_threshold: int = None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setAgeColumn(self, value: str):
        return self._set(age_column=value)

    def setAgeThreshold(self, value: int):
        return self._set(age_threshold=value)

    def setColumnToDrop(self, value: str):
        return self._set(column_to_drop=value)

    def setEmpLengthColumn(self, value: str):
        return self._set(emp_length_column=value)

    def setEmpLengthThreshold(self, value: int):
        return self._set(emp_length_threshold=value)

    def _transform(self, dataframe: DataFrame) -> DataFrame:
        # Check for nulls before dropping them
        null_counts = dataframe.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in dataframe.columns])
        total_nulls = null_counts.rdd.map(lambda row: sum(row)).collect()[0]
        
        if total_nulls > 0:
            dataframe = dataframe.dropna()

        # Filter by age threshold if the column and threshold are set
        age_column = self.getOrDefault(self.age_column)
        age_threshold = self.getOrDefault(self.age_threshold)
        
        if age_column and age_threshold:
            # Cast age column to integer for comparison
            dataframe = dataframe.withColumn(age_column, F.col(age_column).cast(IntegerType()))
            max_age = dataframe.agg(F.max(F.col(age_column))).collect()[0][0]
            if max_age > age_threshold:
                dataframe = dataframe.filter(F.col(age_column) <= age_threshold)
        
        # Filter by employment length first
        emp_length_column = self.getOrDefault(self.emp_length_column)
        emp_length_threshold = self.getOrDefault(self.emp_length_threshold)
        
        if emp_length_column and emp_length_threshold:
            dataframe = dataframe.filter(F.col(emp_length_column) <= emp_length_threshold)

        # Then drop a column and reset index
        column_to_drop = self.getOrDefault(self.column_to_drop)
        if column_to_drop and column_to_drop in dataframe.columns:
            window_spec = Window.orderBy(F.monotonically_increasing_id())
            dataframe = dataframe.drop(column_to_drop)
            dataframe = dataframe.withColumn("index", F.row_number().over(window_spec) - 1)
            dataframe = dataframe.drop("index")
        
        return dataframe


class AgeGroupCategorizer(Transformer, DefaultParamsReadable, DefaultParamsWritable):

    @keyword_only
    def __init__(self, inputCol: str = None, outputCol: str = None, bins: List[int] = None, labels: List[str] = None):
        super(AgeGroupCategorizer, self).__init__()
        self.inputCol = Param(self, "inputCol", "Input column for age values")
        self.outputCol = Param(self, "outputCol", "Output column for age group categories")
        self.bins = Param(self, "bins", "List of bin edges for age groups")
        self.labels = Param(self, "labels", "List of labels for each age group")
        self.setParams(inputCol=inputCol, outputCol=outputCol, bins=bins, labels=labels)

    @keyword_only
    def setParams(self, inputCol: str = None, outputCol: str = None, bins: List[int] = None, labels: List[str] = None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, value: str):
        return self._set(inputCol=value)

    def getInputCol(self):
        return self.getOrDefault(self.inputCol)

    def setOutputCol(self, value: str):
        return self._set(outputCol=value)

    def getOutputCol(self):
        return self.getOrDefault(self.outputCol)

    def setBins(self, value: List[int]):
        return self._set(bins=value)

    def getBins(self):
        return self.getOrDefault(self.bins)

    def setLabels(self, value: List[str]):
        return self._set(labels=value)

    def getLabels(self):
        return self.getOrDefault(self.labels)

    def _transform(self, dataframe: DataFrame) -> DataFrame:
        inputCol = self.getInputCol()
        outputCol = self.getOutputCol()
        bins = self.getBins()
        labels = self.getLabels()

        def assign_age_group(age):
            try:
                age = int(age) 
                for i in range(len(bins) - 1):
                    if bins[i] <= age < bins[i + 1]:
                        return labels[i]
                return labels[-1] 
            except ValueError:
                return labels[-1]  

        age_group_udf = udf(assign_age_group, StringType())
        dataframe = dataframe.withColumn(outputCol, age_group_udf(col(inputCol)))
        
        return dataframe


class IncomeGroupCategorizer(Transformer, DefaultParamsReadable, DefaultParamsWritable):

    inputCol = Param(Params._dummy(), "inputCol", "Input column for income values")
    outputCol = Param(Params._dummy(), "outputCol", "Output column for income group categories")

    @keyword_only
    def __init__(self, inputCol: str = None, outputCol: str = 'income_group'):
        super(IncomeGroupCategorizer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol: str = None, outputCol: str = 'income_group'):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, value: str):
        return self._set(inputCol=value)

    def getInputCol(self):
        return self.getOrDefault(self.inputCol)

    def setOutputCol(self, value: str):
        return self._set(outputCol=value)

    def getOutputCol(self):
        return self.getOrDefault(self.outputCol)

    def _transform(self, dataframe: DataFrame):
        inputCol = self.getInputCol()
        outputCol = self.getOutputCol()

        dataframe = dataframe.withColumn(
            outputCol,
            when(col(inputCol).between(0, 25000), 'low')
            .when(col(inputCol).between(25001, 50000), 'low-middle')
            .when(col(inputCol).between(50001, 75000), 'middle')
            .when(col(inputCol).between(75001, 100000), 'high-middle')
            .otherwise('high')
        )
        return dataframe

class LoanAmountCategorizer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    
    @keyword_only
    def __init__(self, inputCol: str = None, outputCol: str = 'loan_amount_group'):
        super(LoanAmountCategorizer, self).__init__()
        self._inputCol = inputCol
        self._outputCol = outputCol

    @keyword_only
    def setParams(self, inputCol: str = None, outputCol: str = 'loan_amount_group'):
        self._inputCol = inputCol
        self._outputCol = outputCol
        return self

    def setInputCol(self, value: str):
        self._inputCol = value
        return self

    def setOutputCol(self, value: str):
        self._outputCol = value
        return self

    def getInputCol(self):
        return self._inputCol

    def getOutputCol(self):
        return self._outputCol

    def _transform(self, dataframe: DataFrame) -> DataFrame:
        input_col = self.getInputCol()
        output_col = self.getOutputCol()

        dataframe = dataframe.withColumn(
            output_col,
            when(col(input_col).between(0, 5000), 'small')
            .when(col(input_col).between(5001, 10000), 'medium')
            .when(col(input_col).between(10001, 15000), 'high')
            .otherwise('very high')
        )
        
        return dataframe

class RatioFeatureGenerator(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    
    @keyword_only
    def __init__(self):
        super(RatioFeatureGenerator, self).__init__()

    def setParams(self):
        return self

    def _transform(self, dataframe: DataFrame) -> DataFrame:
        dataframe = dataframe.withColumn('loan_to_income_ratio', col('loan_amnt') / col('person_income'))
        dataframe = dataframe.withColumn('loan_to_emp_length_ratio', col('person_emp_length') / col('loan_amnt'))
        dataframe = dataframe.withColumn('int_rate_to_loan_amt_ratio', col('loan_int_rate') / col('loan_amnt'))
        return dataframe

class DataUpsampler:
    def __init__(self, transformed_training_data, target_column):
        """
        Initialize the class with training data and the target column (e.g., 'loan_status').

        :param transformed_training_data: Input PySpark DataFrame
        :param target_column: Column name that contains the target class (e.g., 'loan_status')
        """
        self.transformed_training_data = transformed_training_data
        self.target_column = target_column

    def count_classes(self):
        """
        Count the majority and minority classes in the DataFrame based on the target column.

        :return: Count of majority class, count of minority class, majority class DataFrame, minority class DataFrame
        """
        majority_class = self.transformed_training_data.filter(F.col(self.target_column) == 0)
        minority_class = self.transformed_training_data.filter(F.col(self.target_column) == 1)

        majority_count = majority_class.count()
        minority_count = minority_class.count()

        return majority_count, minority_count, majority_class, minority_class

    def upsample_minority_class(self, majority_count, minority_class):
        """
        Upsample the minority class based on the ratio of majority to minority class counts.

        :param majority_count: Count of the majority class
        :param minority_class: DataFrame containing the minority class
        :return: Upsampled minority class DataFrame
        """
        minority_count = minority_class.count()
        upsample_ratio = majority_count // minority_count

        # Upsample minority class
        upsampled_minority_class = minority_class.withColumn(
            "dummy", F.explode(F.array([F.lit(x) for x in range(upsample_ratio)]))
        ).drop("dummy")

        # Check if additional rows are still needed
        remaining_count = majority_count - upsampled_minority_class.count()

        if remaining_count > 0:
            extra_minority_rows = minority_class.limit(remaining_count)
            final_minority_class = upsampled_minority_class.unionAll(extra_minority_rows)
        else:
            final_minority_class = upsampled_minority_class

        return final_minority_class

    def upsample_data(self):
        """
        Perform upsampling on the minority class and return the upsampled DataFrame.

        :return: Upsampled training DataFrame
        """
        majority_count, minority_count, majority_class, minority_class = self.count_classes()
        final_minority_class = self.upsample_minority_class(majority_count, minority_class)
        upsampled_train_df = majority_class.unionAll(final_minority_class)

        return upsampled_train_df

class CustomTransformer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    
    def __init__(self):
        super(CustomTransformer, self).__init__()

    def _transform(self, dataframe: DataFrame) -> DataFrame:
        dataframe = dataframe.select([F.col(c).alias(c.replace(" ", "_")) for c in dataframe.columns])
        string_columns = [c for c, dtype in dataframe.dtypes if dtype == 'string']

        for col_name in string_columns:
            dataframe = dataframe.withColumn(col_name, col(col_name).cast("float"))

        return dataframe

class CustomVectorAssembler(Transformer, DefaultParamsReadable, DefaultParamsWritable):

    def __init__(self, inputCols=None, outputCol="features"):
        super(CustomVectorAssembler, self).__init__()
        self.inputCols = inputCols
        self.outputCol = outputCol

    def _transform(self, dataframe: DataFrame) -> DataFrame:
        # Assemble features into a single vector
        assembler = VectorAssembler(inputCols=self.inputCols, outputCol=self.outputCol)
        dataframe = assembler.transform(dataframe)
        return dataframe

