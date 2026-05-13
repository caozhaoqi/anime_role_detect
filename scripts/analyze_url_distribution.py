import os

def analyze_url_distribution():
    url_dir = 'spider_image_system/data/img_url'
    url_files = [f for f in os.listdir(url_dir) if f.endswith('_img.txt')]
    
    print('=' * 60)
    print('          URL文件分布情况')
    print('=' * 60)
    
    # 基本统计
    print(f'\n【一、基本统计】')
    print(f'  URL文件总数: {len(url_files)} 个')
    
    # 命名类型分析
    pinyin_count = sum(1 for f in url_files if f.replace('_img.txt', '').islower() and any(c.isdigit() for c in f))
    english_count = sum(1 for f in url_files if f.replace('_img.txt', '')[0].isupper() and not any(c.isdigit() for c in f.replace('_img.txt', '')))
    japanese_count = sum(1 for f in url_files if any('\u3040' <= c <= '\u30ff' for c in f))
    mixed_count = len(url_files) - pinyin_count - english_count - japanese_count
    
    print(f'\n【二、命名类型分布】')
    print(f'  ├── 拼音命名: {pinyin_count} 个')
    print(f'  ├── 英文命名: {english_count} 个')
    print(f'  ├── 日文命名: {japanese_count} 个')
    print(f'  └── 混合命名: {mixed_count} 个')
    
    # URL数量统计
    total_urls = 0
    url_counts = []
    for f in url_files:
        with open(os.path.join(url_dir, f), 'r') as file:
            count = len(file.readlines())
            total_urls += count
            url_counts.append((f, count))
    
    # 排序
    url_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f'\n【三、URL数量统计】')
    print(f'  总URL数量: {total_urls:,} 个')
    print(f'  平均每文件: {total_urls // len(url_files)} 个')
    
    print(f'\n【四、URL数量TOP10】')
    print('-' * 50)
    print(f'{"排名":<4} {"文件":<20} {"URL数量":>8}')
    print('-' * 50)
    for i, (file, count) in enumerate(url_counts[:10], 1):
        print(f'{i:<4} {file[:19]:<20} {count:>8}')
    
    print(f'\n【五、URL数量分布】')
    print('-' * 40)
    ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500), (500, 1000)]
    for r in ranges:
        cnt = sum(1 for _, c in url_counts if r[0] <= c < r[1])
        print(f'  {r[0]}-{r[1]}: {cnt} 个文件')

if __name__ == '__main__':
    analyze_url_distribution()
