# Pandas学习笔记

## 一、Series

- 一维的类似数组的数据结构，可以理解为带有索引的一列数据。与Numpy数组类似，但提供了更多功能。

- 例如：

- ``` python
  import pandas as pd
  s=pd.Series([1,2,3,4,],index=['a','b','c','d'])
  print(s)
  ```

- ```css
  a  1
  b  2
  c  3
  d  4
  ```

## 二、DataFrame

- 二维的类似表格的数据结构，可以理解为带有行和列标签的表，DataFrame就像是Excel或者SQL数据表，非常适合存储结构化数据。

- ```py
  data={'Name':['a','b','c','d'],
        'Age':[15,25,35,40],
        'Salary':[1000,2000,3000,4000]
  }
  df=pd.DataFrame(data)
  print(df)
  ```

## 三、Pandas的主要功能

1. ### 数据的读取和写入

   - 读取和导出

     ```py
     df=pd.read_csv('your_file.csv')
     df.to_csv('output.csv')
     ```

2. ### 数据清洗和处理

   填充缺失值

   ```py
   df.fillna(0,inplace=True)
   ```

   

   删除重复值

   ```py
   df.drop_duplicates(inplace=True)
   ```

   

3. ### 数据筛选与索引

   根据条件筛选

   ```py
   filtered_df=df[df['Age']>30]
   ```

   

   按行或按列索引

   ```py
   selected_columns=df[['Name','Salary']]
   ```

   

4. ### 数据统计与汇总

   计算平均值

   ```py
   df['Salary'].mean()
   ```

   