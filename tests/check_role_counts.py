import json

# 加载索引映射文件
with open('role_index_mapping.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 计算特定角色的条目数量
prana_count = data.count('blda_spider_img_keyword_普拉娜')
arona_count = data.count('blda_spider_img_keyword_阿罗娜')

print(f'blda_spider_img_keyword_普拉娜: {prana_count} 个条目')
print(f'blda_spider_img_keyword_阿罗娜: {arona_count} 个条目')
