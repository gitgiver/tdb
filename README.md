构建知识库
python question2.py --build_kb --data_dir txts --output_dir knowledge_base

执行查询。查询在queries.txt里面编辑，一行为一个查询
python question2.py --batch_queries queries.txt --save_history

