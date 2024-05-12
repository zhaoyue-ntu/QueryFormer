# Exploring the features that are passed to the Input Layer of the QueryFormer Model
In this file, I will walk you through exploring the QueryFormer dataset, which is passed as input to the NN model. The accompanying notebook is [Training Dataset Exploration.ipynb](Training%20Dataset%20Exploration.ipynb). __For generating different code and understanding parts of the code and the output, I took help from GitHub Copilot__. 

## Step 1: Identifying the list of objects that form the training dataset
```python
train_ds = PlanTreeDataset(full_train_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample)
```
In the above line of code a dataset is constructed based on the training query plans. The dataset is an object of class `PlanTreeDataset`, which is defined in [dataset.py](dataset.py).

The constructor of PlanTreeDataset receives as input the following objects:
1. full_train_df
2. encoding
3. hist_file
4. card_norm
5. cost_norm
6. to_predict
7. table_sample

Next I will explore each of these objects. 

## Step 2: Exploring full_train_df
From the notebook section called __Exploring full_train_df__, I have figured out that full_train_df is a Pandas dataframe of shape 2:
```python
full_train_df.shape
```
(10000, 2)

```python
full_train_df.head(2)
```
	id	json
0	0	{"Plan": {"Node Type": "Gather", "Parallel Awa...
1	1	{"Plan": {"Node Type": "Seq Scan", "Parallel A...

The column data types are:
```python
full_train_df.dtypes
```
id       int64
json    object
dtype: object

From the exploration of the full_train_df so far, I figured each element (row) in the dataframe corresponds to query and its query plan, saved as a json string. Json can be used for both nesting and parallel relationships in the query plan tree. 

The following file has a sample json string, representing a query execution plan, in a well format. 

[Output JSON](output.json)

__GitHub Copilot__
This JSON structure represents a query execution plan. It's a hierarchical representation of the steps a database management system takes to respond to a specific query. Here's a breakdown of the structure:

- `Plan`: The root node of the plan. Each plan has a `Node Type` which describes the operation that the node performs. This plan's root node is a `Gather` node, which means it's a parallel query that collects the results from several worker processes.

- `Parallel Aware`: A boolean indicating whether the node can make use of parallel workers.

- `Startup Cost`: An estimate of the cost to start fetching rows.

- `Total Cost`: An estimate of the total cost to fetch all rows.

- `Plan Rows`: The estimated number of rows that the plan will return.

- `Plan Width`: The estimated average width of rows in the plan (in bytes).

- `Actual Startup Time`: The actual time it took to start fetching rows (in milliseconds).

- `Actual Total Time`: The actual total time it took to fetch all rows (in milliseconds).

- `Actual Rows`: The actual number of rows returned by the plan.

- `Actual Loops`: The number of times the node's output was scanned.

- `Workers Planned`: The number of parallel workers planned for this node.

- `Workers Launched`: The number of parallel workers launched for this node.

- `Single Copy`: A boolean indicating whether a single copy of the plan is sufficient.

- `Plans`: An array of sub-plans (children) for this node. Each sub-plan is a similar structure, representing a step in the query execution.

- `Parent Relationship`: Describes the relationship of the current node to its parent.

- `Join Type`: The type of join operation (e.g., "Inner").

- `Inner Unique`: A boolean indicating whether the inner relation of a join is unique.

- `Hash Cond`: The condition for a hash join.

- `Filter`: The filter condition for a scan.

- `Rows Removed by Filter`: The number of rows removed by the filter condition.

- `Hash Buckets`: The number of hash buckets in a hash operation.

- `Original Hash Buckets`: The original number of hash buckets in a hash operation.

- `Hash Batches`: The number of hash batches in a hash operation.

- `Original Hash Batches`: The original number of hash batches in a hash operation.

- `Peak Memory Usage`: The peak memory usage of a hash operation.

- `Planning Time`: The time it took to create the plan (in milliseconds).

- `Execution Time`: The time it took to execute the plan (in milliseconds).

- `Triggers`: An array of triggers for the plan. In this case, there are no triggers.

Each node in the plan represents a step in the execution of the query, and child nodes represent steps that are performed as part of their parent step.

## Step 3: Exploring encoding object
The encoding object is an instance of the class `Encoding`, which is defined in `database_util.py'. 

__GitHub Copilot Explanation Begin__
This Python code defines a class `Encoding` that is used to encode and normalize various types of data, such as filters, joins, tables, and types. Here's a breakdown of the class and its methods:

- `__init__`: This is the constructor method for the class. It initializes several dictionaries used for encoding and decoding data, and sets up mappings between columns, operations, types, joins, and tables and their corresponding indices.

- `normalize_val`: This method normalizes a given value based on the minimum and maximum values of the column it belongs to. Normalization is a common preprocessing step in machine learning and data analysis to ensure that all data is on a similar scale.

- `encode_filters`: This method encodes a list of filters. Each filter is a dictionary, and the method returns a dictionary with the encoded column IDs, operation IDs, and normalized values.

- `encode_join`: This method encodes a join operation. If the join operation has not been seen before, it is added to the `join2idx` and `idx2join` dictionaries.

- `encode_table`: This method encodes a table name. If the table name has not been seen before, it is added to the `table2idx` and `idx2table` dictionaries.

- `encode_type`: This method encodes a node type. If the node type has not been seen before, it is added to the `type2idx` and `idx2type` dictionaries.

The purpose of this class is to provide a consistent way to encode and decode various types of data, which can be useful in many contexts, such as when working with databases, performing data analysis, or training machine learning models.
__GitHub Copilot Explanation End__

The cnoding object is stored in a file as a serialized object. It's loaded from the file and deserialized. From the 
The following line of code loads this object from the file into the Python runtime:
```python
encoding_ckpt = torch.load('checkpoints/encoding.pt')
```

`encoding_ckpt` is a dictionary. It has only one key: `encoding`.
```python
encoding_ckpt.keys()
```
dict_keys(['encoding'])

The authors of QueryFormer had created an instance object of the Encoding class and stored it in a checkpoint file (.pt) extension. Now the above load 
command loads that stored object (serialized object) from the file into Python runtime. When serialzing an object, all the values of its member variables 
are stored. This Encoding class has the following member variables. Using GitHub Copilot, I explained them as follows:

__GitHub Copilot Explanation Begin__
When you serialize an object of the `Encoding` class and store it into a checkpoint file, all the instance variables (attributes) of the object are stored. These include:

- `column_min_max_vals`: A dictionary that stores the minimum and maximum values for each column.
- `col2idx`: A dictionary that maps column names to indices.
- `op2idx`: A dictionary that maps operators to indices.
- `idx2col`: A dictionary that maps indices to column names.
- `idx2op`: A dictionary that maps indices to operators.
- `type2idx`: A dictionary that maps types to indices.
- `idx2type`: A dictionary that maps indices to types.
- `join2idx`: A dictionary that maps joins to indices.
- `idx2join`: A dictionary that maps indices to joins.
- `table2idx`: A dictionary that maps table names to indices.
- `idx2table`: A dictionary that maps indices to table names.

These attributes are stored in the checkpoint file in a format that allows them to be loaded back into a Python program and used to recreate an identical `Encoding` object.

Note that the actual methods of the class (`normalize_val`, `encode_filters`, `encode_join`, `encode_table`, `encode_type`) are not stored in the checkpoint file. Only the data (the state of the object) is stored. When you load the checkpoint, you'll need to have the class definition available in your program to use the loaded data.
__GitHub Copilot Explanation End__

Now I will print the value of each member variable in the Encoding class for the loaded instance of Encoding object. 
1. printing `column_min_max_vals`
```python
print(encoding.column_min_max_vals)
```
{'t.id': [1.0, 2528312.0], 't.kind_id': [1.0, 7.0], 't.production_year': [1880.0, 2019.0], 'mc.id': [1.0, 2609129.0], 'mc.company_id': [1.0, 234997.0], 'mc.movie_id': [2.0, 2525745.0], 'mc.company_type_id': [1.0, 2.0], 'ci.id': [1.0, 36244344.0], 'ci.movie_id': [1.0, 2525975.0], 'ci.person_id': [1.0, 4061926.0], 'ci.role_id': [1.0, 11.0], 'mi.id': [1.0, 14835720.0], 'mi.movie_id': [1.0, 2526430.0], 'mi.info_type_id': [1.0, 110.0], 'mi_idx.id': [1.0, 1380035.0], 'mi_idx.movie_id': [2.0, 2525793.0], 'mi_idx.info_type_id': [99.0, 113.0], 'mk.id': [1.0, 4523930.0], 'mk.movie_id': [2.0, 2525971.0], 'mk.keyword_id': [1.0, 134170.0]}


__GitHub Copilot Begin__
The `column_min_max_vals` dictionary you've printed out is storing the minimum and maximum values for each column in your dataset. Each key in the dictionary is a column name, and the value associated with each key is a list of two numbers: the minimum value and the maximum value for that column.

Here's a breakdown of a few entries:

- `'t.id': [1.0, 2528312.0]`: For the column `t.id`, the minimum value is `1.0` and the maximum value is `2528312.0`.
- `'t.kind_id': [1.0, 7.0]`: For the column `t.kind_id`, the minimum value is `1.0` and the maximum value is `7.0`.
- `'t.production_year': [1880.0, 2019.0]`: For the column `t.production_year`, the minimum value is `1880.0` and the maximum value is `2019.0`.

This kind of information is often used in data preprocessing to normalize data, i.e., to scale numeric values to a standard range, such as 0-1.
__GitHub Copilot End__

2. printing `col2idx`
It's a dictionary which has the label encoding for each unique column in the dataset.

```python
print(encoding.col2idx)
```
{'t.id': 0, 't.kind_id': 1, 't.production_year': 2, 'mc.id': 3, 'mc.company_id': 4, 'mc.movie_id': 5, 'mc.company_type_id': 6, 'ci.id': 7, 'ci.movie_id': 8, 'ci.person_id': 9, 'ci.role_id': 10, 'mi.id': 11, 'mi.movie_id': 12, 'mi.info_type_id': 13, 'mi_idx.id': 14, 'mi_idx.movie_id': 15, 'mi_idx.info_type_id': 16, 'mk.id': 17, 'mk.movie_id': 18, 'mk.keyword_id': 19, 'NA': 20}

3. printing `op2idx`
```python
print(encoding.op2idx)
```
{'>': 0, '=': 1, '<': 2, 'NA': 3}

__GitHub Copilot Begin__
The `op2idx` dictionary is mapping operators to indices. This is often done in machine learning tasks to convert categorical data into numerical data that can be processed by a model. 

Here's what each entry means:

- `'>'`: The greater than operator is mapped to the index `0`.
- `'='`: The equals operator is mapped to the index `1`.
- `'<'`: The less than operator is mapped to the index `2`.
- `'NA'`: 'NA' (which likely stands for 'Not Available' or 'Not Applicable') is mapped to the index `3`.

So, if you have a filter operation in your data represented by these operators, you can convert them into their corresponding numerical indices for further processing.
__GitHub Copilot End__

4. printing `idx2col`
```python
print(encoding.idx2col)
```

This instance variable, idx2col, has the inverse mapping between label coding and column. I.e., it's opposite to col2idx, where you can pass a column 
name and get back its index (label encoding). Using idx2col dictionary, you can pass a label encoding and get back the column name. 
__I am not sure yet where it's needed, though.__

5. printing `type2idx`
This dictionary has the label encoding for each unique database operator in the query execution plans. Here are a few sample key-value pairs from this dictionary:

'Hash Join': 1
'Index Scan': 7


