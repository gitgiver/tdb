命令行使用：

构建知识库
python question2.py --build_kb --data_dir txts --output_dir knowledge_base

执行查询。查询在queries.txt里面编辑，一行为一个查询
python question2.py --batch_queries queries.txt --save_history

需要先在queries.txt里面写好你要问的问题，一行一个问题。

对于知识库更新功能，也就是问题3
命令是python question3.py  --data_dir="txts" --initial_scan 

===================================================

web界面使用：

运行web界面：
1.命令行执行python app.py
2.浏览器输入链接
框架为flask

