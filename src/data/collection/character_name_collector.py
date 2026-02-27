#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色名称采集器

优化角色名称的采集和验证策略
"""

import os
import re
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('character_name_collector')


class CharacterNameCollector:
    """
    角色名称采集器
    负责从角色文件中提取和验证角色名称
    """
    
    def __init__(self):
        """
        初始化角色名称采集器
        """
        # 常见的非角色关键词
        self.non_character_keywords = [
            # 版本和活动
            '版本', '活动', '更新', '时间', '联动', '合作', '主题', '特别篇', '系列',
            '年', '月', '日', '小时', '分钟', '周年', '庆典', '节日',
            
            # 游戏系统
            '游戏类型', '货币', '消耗品', '强化材料', '遗器', '光锥', '任务道具', '其他材料',
            '系统', '机制', '玩法', '功能', '模式', '界面', '设置',
            
            # 地理和地点
            '地区', '城市', '国家', '地图', '区域', '地点', '场景', '副本',
            
            # 组织和势力
            '组织', '势力', '阵营', '团队', '公会', '家族', '党派',
            
            # 其他非角色词汇
            '参考资料', '来源', '注', '附录', '索引', '目录', '表格',
            '统计', '数据', '分析', '报告', '指南', '攻略', '教程',
            '技巧', '提示', '建议', '心得', '经验', '分享', '讨论',
            '问题', '解答', 'FAQ', '常见问题', '注意事项', '警告',
            '错误', '修复', '更新日志', '变更记录', '历史', '回顾',
            '展望', '未来', '计划', '预告', '预览', '测试', 'beta',
            'alpha', 'demo', '试玩', '体验', '反馈', '评价', '评分',
            '排名', '榜单', '人气', '热度', '关注', '讨论度', '话题',
            '趋势', '流行', '热门', '冷门', '稀有', '常见', '普通',
            '特殊', '限定', '绝版', '复刻', '返场', '上新', '推出',
            '发布', '上线', '公测', '内测', '开服', '关服', '停服',
            '维护', '更新', '升级', '优化', '修复', '调整', '平衡',
            '改动', '变更', '修改', '删除', '添加', '增加', '减少',
            '提升', '降低', '增强', '削弱', '强化', '弱化', '进化',
            '退化', '解锁', '锁定', '激活', '禁用', '开启', '关闭',
            '开启条件', '解锁条件', '使用条件', '获取条件', '达成条件',
            '任务', '剧情', '故事', '背景', '世界观', '设定', '设定集',
            '资料集', '画集', '图集', '音乐', '音效', '配音', '声优',
            'CV', '演员', '配音演员', '歌手', '作曲家', '音乐人', '艺术家',
            '设计师', '策划', '制作', '开发', '团队', '公司', '厂商',
            '发行', '运营', '代理', '版权', '商标', '专利', '授权',
            '合作', '联动', '联名', '主题', '限定', '特别', '纪念',
            '周年', '生日', '节日', '庆典', '活动', '赛事', '比赛',
            '竞争', '排名', '奖励', '奖品', '礼品', '福利', '优惠',
            '折扣', '促销', '销售', '购买', '充值', '消费', '花费',
            '收入', '收益', '利润', '成本', '价格', '价值', '性价比',
            '性能', '配置', '要求', '推荐', '最低', '最高', '系统',
            '硬件', '软件', '设备', '平台', '终端', '客户端', '服务器',
            '网络', '连接', '延迟', '卡顿', '崩溃', '错误', 'bug',
            '修复', '解决', '处理', '应对', '预防', '避免', '注意',
            '提醒', '警告', '提示', '建议', '指导', '教程', '攻略',
            '技巧', '方法', '步骤', '流程', '顺序', '时间', '地点',
            '人物', '道具', '装备', '武器', '防具', '饰品', '配件',
            '材料', '资源', '货币', '金币', '钻石', '宝石', '点券',
            '积分', '经验', '等级', '技能', '天赋', '属性', '数值',
            '伤害', '治疗', '防御', '攻击', '速度', '暴击', '命中',
            '闪避', '抵抗', '穿透', '吸血', '回蓝', '冷却', '范围',
            '距离', '持续', '间隔', '频率', '概率', '几率', '成功率',
            '失败率', '效率', '效果', '作用', '影响', '意义', '价值',
            '重要性', '必要性', '可选性', '必须', '可选', '推荐', '不推荐',
            '优先', '次优', '最佳', '最差', '最强', '最弱', '最好', '最坏',
            '优点', '缺点', '优势', '劣势', '长处', '短处', '好处', '坏处',
            '利', '弊', '益', '害', '得', '失', '成', '败', '功', '过',
            '是', '非', '对', '错', '好', '坏', '善', '恶', '美', '丑',
            '真', '假', '实', '虚', '有', '无', '存在', '不存在',
            '可能', '不可能', '可行', '不可行', '有效', '无效',
            '有用', '无用', '适用', '不适用', '适合', '不适合',
            '符合', '不符合', '满足', '不满足', '达到', '未达到',
            '完成', '未完成', '成功', '失败', '通过', '未通过',
            '合格', '不合格', '优秀', '良好', '中等', '及格', '不及格',
            '高', '低', '大', '小', '多', '少', '长', '短', '宽', '窄',
            '厚', '薄', '重', '轻', '强', '弱', '快', '慢', '好', '坏',
            '优', '劣', '佳', '差', '良', '莠', '精', '粗', '细', '疏',
            '密', '浓', '淡', '深', '浅', '高', '矮', '胖', '瘦', '美', '丑',
            '帅', '丑', '可爱', '不可爱', '漂亮', '丑陋', '英俊', '难看',
            '聪明', '愚蠢', '善良', '邪恶', '勇敢', '懦弱', '坚强', '脆弱',
            '乐观', '悲观', '积极', '消极', '主动', '被动', '热情', '冷漠',
            '友好', '敌对', '善良', '恶毒', '正直', '邪恶', '诚实', '虚伪',
            '守信', '失信', '忠诚', '背叛', '勤劳', '懒惰', '节俭', '浪费',
            '谦虚', '骄傲', '礼貌', '粗鲁', '文明', '野蛮', '理性', '感性',
            '冷静', '冲动', '稳重', '轻浮', '成熟', '幼稚', '自信', '自卑',
            '自尊', '自贱', '自爱', '自弃', '自强', '自弱', '自立', '依赖',
            '自主', '顺从', '自由', '束缚', '开放', '保守', '创新', '守旧',
            '进步', '落后', '发展', '停滞', '前进', '后退', '上升', '下降',
            '增长', '减少', '增加', '降低', '提高', '降低', '上升', '下降',
            '增强', '减弱', '扩大', '缩小', '扩张', '收缩', '膨胀', '收缩',
            '繁荣', '衰退', '兴盛', '衰落', '强大', '弱小', '富裕', '贫穷',
            '幸福', '痛苦', '快乐', '悲伤', '喜悦', '悲伤', '高兴', '难过',
            '兴奋', '沮丧', '激动', '平静', '紧张', '放松', '焦虑', '安心',
            '担心', '放心', '恐惧', '勇敢', '害怕', '勇敢', '惊慌', '镇定',
            '愤怒', '平静', '生气', '冷静', '仇恨', '宽容', '爱', '恨',
            '喜欢', '讨厌', '欣赏', '鄙视', '尊重', '轻视', '重视', '忽视',
            '关心', '冷漠', '爱护', '伤害', '保护', '破坏', '建设', '摧毁',
            '创造', '毁灭', '生产', '消耗', '制造', '破坏', '构建', '拆解',
            '组合', '分离', '整合', '分裂', '统一', '分裂', '团结', '分裂',
            '合作', '对抗', '协作', '竞争', '配合', '冲突', '和谐', '矛盾',
            '和平', '战争', '友好', '敌对', '同盟', '敌对', '伙伴', '敌人',
            '朋友', '敌人', '亲人', '外人', '家人', '外人', '熟人', '陌生人',
            '同事', '对手', '队友', '敌人', '伙伴', '敌人', '支持者', '反对者',
            '粉丝', '黑粉', '观众', '听众', '读者', '用户', '玩家', '客户',
            '消费者', '生产者', '供应者', '需求者', '卖家', '买家', '商家', '顾客',
            '服务者', '被服务者', '提供者', '接受者', '给予者', '索取者',
            '贡献者', '受益者', '创造者', '享受者', '牺牲者', '幸存者',
            '胜利者', '失败者', '成功者', '失败者', '英雄', '反派', '主角', '配角',
            '正面', '反面', '主要', '次要', '核心', '边缘', '重要', '不重要',
            '关键', '非关键', '必要', '不必要', '必须', '不必须', '需要', '不需要',
            '应该', '不应该', '可以', '不可以', '允许', '不允许', '同意', '不同意',
            '支持', '反对', '赞成', '反对', '肯定', '否定', '认可', '否定',
            '接受', '拒绝', '欢迎', '排斥', '包容', '排斥', '理解', '误解',
            '明白', '困惑', '清楚', '模糊', '明确', '模糊', '确定', '不确定',
            '肯定', '怀疑', '相信', '怀疑', '信任', '怀疑', '信心', '怀疑',
            '希望', '绝望', '期待', '失望', '盼望', '失望', '向往', '厌恶',
            '追求', '逃避', '喜欢', '逃避', '愿意', '不愿意', '乐意', '不乐意',
            '主动', '被动', '积极', '消极', '乐观', '悲观', '向上', '向下',
            '正面', '负面', '积极', '消极', '健康', '不健康', '有益', '有害',
            '好的', '坏的', '正确的', '错误的', '合理的', '不合理的',
            '适当的', '不适当的', '合适的', '不合适的', '恰当的', '不恰当的',
            '符合的', '不符合的', '满足的', '不满足的', '达到的', '未达到的',
            '完成的', '未完成的', '成功的', '失败的', '通过的', '未通过的',
            '合格的', '不合格的', '优秀的', '良好的', '中等的', '及格的', '不及格的',
        ]
        
        # 常见的角色后缀和前缀
        self.character_prefixes = ['', '']  # 暂时为空，可根据需要添加
        self.character_suffixes = ['', '']  # 暂时为空，可根据需要添加
        
        # 常见的角色类型词汇
        self.character_type_keywords = [
            '角色', '人物', '主角', '配角', '主人公', '配角', '人物', '角色',
            '英雄', '反派', '正派', '反派', '正面角色', '反面角色', '主要角色', '次要角色',
            '核心角色', '边缘角色', '重要角色', '不重要角色', '关键角色', '非关键角色',
            '必要角色', '不必要角色', '必须角色', '不必须角色', '需要角色', '不需要角色',
            '应该角色', '不应该角色', '可以角色', '不可以角色', '允许角色', '不允许角色',
            '同意角色', '不同意角色', '支持角色', '反对角色', '赞成角色', '反对角色',
            '肯定角色', '否定角色', '认可角色', '否定角色', '接受角色', '拒绝角色',
            '欢迎角色', '排斥角色', '包容角色', '排斥角色', '理解角色', '误解角色',
            '明白角色', '困惑角色', '清楚角色', '模糊角色', '明确角色', '模糊角色',
            '确定角色', '不确定角色', '肯定角色', '怀疑角色', '相信角色', '怀疑角色',
            '信任角色', '怀疑角色', '信心角色', '怀疑角色', '希望角色', '绝望角色',
            '期待角色', '失望角色', '盼望角色', '失望角色', '向往角色', '厌恶角色',
            '追求角色', '逃避角色', '喜欢角色', '逃避角色', '愿意角色', '不愿意角色',
            '乐意角色', '不乐意角色', '主动角色', '被动角色', '积极角色', '消极角色',
            '乐观角色', '悲观角色', '向上角色', '向下角色', '正面角色', '负面角色',
            '积极角色', '消极角色', '健康角色', '不健康角色', '有益角色', '有害角色',
            '好的角色', '坏的角色', '正确的角色', '错误的角色', '合理的角色', '不合理的角色',
            '适当的角色', '不适当的角色', '合适的角色', '不合适的角色', '恰当的角色', '不恰当的角色',
            '符合的角色', '不符合的角色', '满足的角色', '不满足的角色', '达到的角色', '未达到的角色',
            '完成的角色', '未完成的角色', '成功的角色', '失败的角色', '通过的角色', '未通过的角色',
            '合格的角色', '不合格的角色', '优秀的角色', '良好的角色', '中等的角色', '及格的角色', '不及格的角色',
        ]
    
    def is_character_name(self, text):
        """
        判断文本是否为角色名称
        
        Args:
            text: 待判断的文本
            
        Returns:
            bool: 是否为角色名称
        """
        # 过滤空字符串
        if not text or not text.strip():
            return False
        
        # 过滤太短或太长的文本
        text_stripped = text.strip()
        if len(text_stripped) < 2 or len(text_stripped) > 30:
            return False
        
        # 过滤包含非角色关键词的文本
        for keyword in self.non_character_keywords:
            if keyword in text_stripped:
                return False
        
        # 过滤纯数字
        if text_stripped.isdigit():
            return False
        
        # 过滤纯符号
        if not any(c.isalnum() for c in text_stripped):
            return False
        
        # 过滤包含版本号格式的文本
        version_pattern = r'\d+\.\d+'
        if re.search(version_pattern, text_stripped):
            return False
        
        # 过滤包含日期格式的文本
        date_pattern = r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}'
        if re.search(date_pattern, text_stripped):
            return False
        
        # 过滤包含时间格式的文本
        time_pattern = r'\d{1,2}[:：]\d{1,2}(:：\d{1,2})?'
        if re.search(time_pattern, text_stripped):
            return False
        
        # 过滤包含特殊符号的文本（保留常见的角色名称符号）
        allowed_symbols = ['·', '•', '＆', '&', ' ', '-', '_', '～', '—', '─']
        for char in text_stripped:
            if not char.isalnum() and char not in allowed_symbols:
                return False
        
        # 过滤包含网址的文本
        url_pattern = r'https?://\S+'
        if re.search(url_pattern, text_stripped):
            return False
        
        # 过滤包含邮箱的文本
        email_pattern = r'\S+@\S+\.\S+'
        if re.search(email_pattern, text_stripped):
            return False
        
        # 过滤包含文件路径的文本
        path_pattern = r'[a-zA-Z]:\\|/'
        if re.search(path_pattern, text_stripped):
            return False
        
        # 过滤包含命令的文本
        command_pattern = r'^\w+\s+-'
        if re.search(command_pattern, text_stripped):
            return False
        
        # 过滤包含代码的文本
        code_pattern = r'\{\s*\}|\[\s*\]|\(\s*\)|<\s*>'
        if re.search(code_pattern, text_stripped):
            return False
        
        # 过滤包含标签的文本
        tag_pattern = r'<[^>]+>'
        if re.search(tag_pattern, text_stripped):
            return False
        
        # 过滤包含括号的文本（保留常见的角色名称括号）
        bracket_pattern = r'[\(\)\[\]\{\}\【\】\『\』\「\」\《\》]'
        if re.search(bracket_pattern, text_stripped):
            return False
        
        # 过滤包含表情符号的文本
        emoji_pattern = r'[\u2600-\u27BF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|\uD83E[\uDD00-\uDDFF]'
        if re.search(emoji_pattern, text_stripped):
            return False
        
        # 过滤包含特殊字符的文本
        special_char_pattern = r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]'
        if re.search(special_char_pattern, text_stripped):
            return False
        
        # 过滤包含英文单词的文本（保留常见的角色名称英文）
        english_word_pattern = r'[a-zA-Z]{4,}'
        if re.search(english_word_pattern, text_stripped):
            return False
        
        # 过滤包含数字和字母混合的文本
        alpha_num_pattern = r'[a-zA-Z]+\d+|\d+[a-zA-Z]+'
        if re.search(alpha_num_pattern, text_stripped):
            return False
        
        # 过滤包含重复字符的文本
        repeat_pattern = r'(.)\1{3,}'
        if re.search(repeat_pattern, text_stripped):
            return False
        
        # 过滤包含空格过多的文本
        if text_stripped.count(' ') > 3:
            return False
        
        # 过滤包含连字符过多的文本
        if text_stripped.count('-') > 2:
            return False
        
        # 过滤包含点过多的文本
        if text_stripped.count('.') > 2:
            return False
        
        # 过滤包含冒号的文本
        if ':' in text_stripped or '：' in text_stripped:
            return False
        
        # 过滤包含分号的文本
        if ';' in text_stripped or '；' in text_stripped:
            return False
        
        # 过滤包含逗号的文本
        if ',' in text_stripped or '，' in text_stripped:
            return False
        
        # 过滤包含句号的文本
        if '.' in text_stripped or '。' in text_stripped:
            return False
        
        # 过滤包含问号的文本
        if '?' in text_stripped or '？' in text_stripped:
            return False
        
        # 过滤包含感叹号的文本
        if '!' in text_stripped or '！' in text_stripped:
            return False
        
        # 过滤包含引号的文本
        if '"' in text_stripped or "'" in text_stripped or '“' in text_stripped or '”' in text_stripped:
            return False
        
        # 过滤包含撇号的文本
        if '`' in text_stripped or '’' in text_stripped:
            return False
        
        # 过滤包含反斜杠的文本
        if '\\' in text_stripped:
            return False
        
        # 过滤包含正斜杠的文本
        if '/' in text_stripped:
            return False
        
        # 过滤包含竖线的文本
        if '|' in text_stripped:
            return False
        
        # 过滤包含井号的文本
        if '#' in text_stripped:
            return False
        
        # 过滤包含美元符号的文本
        if '$' in text_stripped:
            return False
        
        # 过滤包含百分号的文本
        if '%' in text_stripped:
            return False
        
        # 过滤包含插入符号的文本
        if '^' in text_stripped:
            return False
        
        # 过滤包含和号的文本
        if '&' in text_stripped:
            return False
        
        # 过滤包含星号的文本
        if '*' in text_stripped:
            return False
        
        # 过滤包含左圆括号的文本
        if '(' in text_stripped:
            return False
        
        # 过滤包含右圆括号的文本
        if ')' in text_stripped:
            return False
        
        # 过滤包含左方括号的文本
        if '[' in text_stripped:
            return False
        
        # 过滤包含右方括号的文本
        if ']' in text_stripped:
            return False
        
        # 过滤包含左花括号的文本
        if '{' in text_stripped:
            return False
        
        # 过滤包含右花括号的文本
        if '}' in text_stripped:
            return False
        
        # 过滤包含左尖括号的文本
        if '<' in text_stripped:
            return False
        
        # 过滤包含右尖括号的文本
        if '>' in text_stripped:
            return False
        
        # 过滤包含波浪号的文本
        if '~' in text_stripped:
            return False
        
        # 过滤包含反引号的文本
        if '`' in text_stripped:
            return False
        
        # 过滤包含等于号的文本
        if '=' in text_stripped:
            return False
        
        # 过滤包含加号的文本
        if '+' in text_stripped:
            return False
        
        # 过滤包含下划线的文本
        if '_' in text_stripped:
            return False
        
        # 过滤包含减号的文本
        if '-' in text_stripped:
            return False
        
        # 过滤包含分音符的文本
        if '¨' in text_stripped:
            return False
        
        # 过滤包含重音符号的文本
        if '´' in text_stripped or '`' in text_stripped or '^' in text_stripped or '~' in text_stripped:
            return False
        
        # 过滤包含特殊空格的文本
        if '\u00A0' in text_stripped:
            return False
        
        # 过滤包含零宽字符的文本
        if '\u200B' in text_stripped or '\u200C' in text_stripped or '\u200D' in text_stripped:
            return False
        
        # 过滤包含控制字符的文本
        control_char_pattern = r'[\x00-\x1F\x7F]'
        if re.search(control_char_pattern, text_stripped):
            return False
        
        # 过滤包含不可打印字符的文本
        if any(ord(c) < 32 and c not in ' \t\n\r\f\v' for c in text_stripped):
            return False
        
        # 过滤包含全角字符的文本（保留全角中文）
        full_width_pattern = r'[\uff01-\uff5e]'
        if re.search(full_width_pattern, text_stripped):
            # 检查是否包含全角字母或数字
            full_width_alpha_num = r'[ａ-ｚＡ-Ｚ０-９]'
            if re.search(full_width_alpha_num, text_stripped):
                return False
        
        # 过滤包含半角字符的文本（保留半角中文）
        half_width_pattern = r'[\x20-\x7E]'
        if re.search(half_width_pattern, text_stripped):
            # 检查是否包含半角字母或数字
            half_width_alpha_num = r'[a-zA-Z0-9]'
            if re.search(half_width_alpha_num, text_stripped):
                return False
        
        # 过滤包含混合大小写的文本
        if any(c.isupper() for c in text_stripped) and any(c.islower() for c in text_stripped):
            return False
        
        # 过滤包含混合全半角的文本
        has_full_width = any(ord(c) > 127 for c in text_stripped)
        has_half_width = any(ord(c) <= 127 for c in text_stripped)
        if has_full_width and has_half_width:
            return False
        
        # 过滤包含多余空格的文本
        if '  ' in text_stripped:
            return False
        
        # 过滤包含首尾空格的文本
        if text_stripped != text_stripped.strip():
            return False
        
        # 过滤包含特殊格式的文本
        special_format_pattern = r'[*_~`#]'
        if re.search(special_format_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown格式的文本
        markdown_pattern = r'^#{1,6}\s|^-\s|^\d+\.\s|^>\s|^```'
        if re.search(markdown_pattern, text_stripped):
            return False
        
        # 过滤包含HTML格式的文本
        html_pattern = r'<[^>]+>'
        if re.search(html_pattern, text_stripped):
            return False
        
        # 过滤包含XML格式的文本
        xml_pattern = r'<[^>]+>'
        if re.search(xml_pattern, text_stripped):
            return False
        
        # 过滤包含JSON格式的文本
        json_pattern = r'\{\s*"|\[\s*"'
        if re.search(json_pattern, text_stripped):
            return False
        
        # 过滤包含YAML格式的文本
        yaml_pattern = r'^\w+:\s|^-\s'
        if re.search(yaml_pattern, text_stripped):
            return False
        
        # 过滤包含INI格式的文本
        ini_pattern = r'^\[\w+\]'
        if re.search(ini_pattern, text_stripped):
            return False
        
        # 过滤包含配置文件格式的文本
        config_pattern = r'^\w+\s*='
        if re.search(config_pattern, text_stripped):
            return False
        
        # 过滤包含日志格式的文本
        log_pattern = r'^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}'
        if re.search(log_pattern, text_stripped):
            return False
        
        # 过滤包含时间戳的文本
        timestamp_pattern = r'^\d{10}$|^\d{13}$'
        if re.search(timestamp_pattern, text_stripped):
            return False
        
        # 过滤包含UUID的文本
        uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        if re.search(uuid_pattern, text_stripped):
            return False
        
        # 过滤包含MAC地址的文本
        mac_pattern = r'^([0-9a-fA-F]{2}[:-]){5}([0-9a-fA-F]{2})$'
        if re.search(mac_pattern, text_stripped):
            return False
        
        # 过滤包含IP地址的文本
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        if re.search(ip_pattern, text_stripped):
            return False
        
        # 过滤包含IPv6地址的文本
        ipv6_pattern = r'^([0-9a-fA-F]{0,4}:){1,7}([0-9a-fA-F]{0,4})$'
        if re.search(ipv6_pattern, text_stripped):
            return False
        
        # 过滤包含URL路径的文本
        url_path_pattern = r'^/\S+'
        if re.search(url_path_pattern, text_stripped):
            return False
        
        # 过滤包含文件扩展名的文本
        file_ext_pattern = r'\.\w{1,5}$'
        if re.search(file_ext_pattern, text_stripped):
            return False
        
        # 过滤包含命令行参数的文本
        cli_pattern = r'^-\w+|^--\w+'
        if re.search(cli_pattern, text_stripped):
            return False
        
        # 过滤包含环境变量的文本
        env_pattern = r'^\$\w+'
        if re.search(env_pattern, text_stripped):
            return False
        
        # 过滤包含正则表达式的文本
        regex_pattern = r'^/.*?/'
        if re.search(regex_pattern, text_stripped):
            return False
        
        # 过滤包含代码注释的文本
        comment_pattern = r'^//|^#|^/\*|^\*/'
        if re.search(comment_pattern, text_stripped):
            return False
        
        # 过滤包含SQL语句的文本
        sql_pattern = r'^(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE|GRANT|REVOKE)\s'
        if re.search(sql_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Shell命令的文本
        shell_pattern = r'^\w+\s*\|\s*\w+|^\w+\s*;\s*\w+'
        if re.search(shell_pattern, text_stripped):
            return False
        
        # 过滤包含PowerShell命令的文本
        powershell_pattern = r'^\w+\s*\|\|\s*\w+|^\w+\s*&&\s*\w+'
        if re.search(powershell_pattern, text_stripped):
            return False
        
        # 过滤包含Python代码的文本
        python_pattern = r'^(import|from|def|class|if|elif|else|for|while|try|except|finally|with|as|pass|break|continue|return|yield|raise|assert|del|global|nonlocal|lambda|and|or|not|in|is|True|False|None)\s'
        if re.search(python_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含JavaScript代码的文本
        javascript_pattern = r'^(var|let|const|function|class|if|else|for|while|do|switch|case|default|try|catch|finally|with|return|break|continue|throw|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(javascript_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Java代码的文本
        java_pattern = r'^(public|private|protected|static|final|abstract|synchronized|native|transient|volatile|class|interface|enum|package|import|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(java_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含C/C++代码的文本
        cpp_pattern = r'^(public|private|protected|static|final|abstract|synchronized|native|transient|volatile|class|struct|union|enum|namespace|using|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(cpp_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含C#代码的文本
        csharp_pattern = r'^(public|private|protected|static|final|abstract|synchronized|native|transient|volatile|class|struct|union|enum|namespace|using|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(csharp_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Go代码的文本
        go_pattern = r'^(package|import|func|var|const|type|struct|interface|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(go_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Rust代码的文本
        rust_pattern = r'^(fn|let|const|static|struct|enum|trait|impl|mod|use|if|else|for|while|loop|match|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(rust_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含PHP代码的文本
        php_pattern = r'^(<?php|function|class|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(php_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Ruby代码的文本
        ruby_pattern = r'^(def|class|module|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(ruby_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Swift代码的文本
        swift_pattern = r'^(func|let|var|class|struct|enum|protocol|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(swift_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Kotlin代码的文本
        kotlin_pattern = r'^(fun|val|var|class|interface|enum|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(kotlin_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Dart代码的文本
        dart_pattern = r'^(void|int|double|String|bool|List|Map|Set|Function|class|interface|enum|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(dart_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含TypeScript代码的文本
        typescript_pattern = r'^(void|int|double|string|boolean|number|any|unknown|never|object|array|tuple|interface|type|class|enum|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(typescript_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含HTML标签的文本
        html_tag_pattern = r'<[^>]+>'
        if re.search(html_tag_pattern, text_stripped):
            return False
        
        # 过滤包含XML标签的文本
        xml_tag_pattern = r'<[^>]+>'
        if re.search(xml_tag_pattern, text_stripped):
            return False
        
        # 过滤包含CSS选择器的文本
        css_selector_pattern = r'^\w+\s*{|^#\w+|^\.\w+'
        if re.search(css_selector_pattern, text_stripped):
            return False
        
        # 过滤包含正则表达式的文本
        regex_pattern = r'^/.*?/'
        if re.search(regex_pattern, text_stripped):
            return False
        
        # 过滤包含数学公式的文本
        math_pattern = r'^\$\$.*?\$\$|^\$.*?\$'
        if re.search(math_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown链接的文本
        markdown_link_pattern = r'\[.*?\]\(.*?\)'
        if re.search(markdown_link_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown图片的文本
        markdown_image_pattern = r'!\[.*?\]\(.*?\)'
        if re.search(markdown_image_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown标题的文本
        markdown_heading_pattern = r'^#{1,6}\s'
        if re.search(markdown_heading_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown列表的文本
        markdown_list_pattern = r'^-\s|^\d+\.\s'
        if re.search(markdown_list_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown引用的文本
        markdown_quote_pattern = r'^>\s'
        if re.search(markdown_quote_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown代码块的文本
        markdown_code_pattern = r'^```'
        if re.search(markdown_code_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown粗体的文本
        markdown_bold_pattern = r'\*\*.*?\*\*|__.*?__'
        if re.search(markdown_bold_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown斜体的文本
        markdown_italic_pattern = r'\*.*?\*|_.*?_' 
        if re.search(markdown_italic_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown删除线的文本
        markdown_strikethrough_pattern = r'~~.*?~~'
        if re.search(markdown_strikethrough_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown行内代码的文本
        markdown_inline_code_pattern = r'`.*?`'
        if re.search(markdown_inline_code_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown表格的文本
        markdown_table_pattern = r'^\|.*?\|'
        if re.search(markdown_table_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown分隔线的文本
        markdown_hr_pattern = r'^-{3,}|^\*{3,}|^_{3,}'
        if re.search(markdown_hr_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown脚注的文本
        markdown_footnote_pattern = r'\[\^.*?\]'
        if re.search(markdown_footnote_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown定义列表的文本
        markdown_definition_pattern = r'^\w+\s*:'
        if re.search(markdown_definition_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown任务列表的文本
        markdown_task_pattern = r'^-\s*\[ \]|^-\s*\[x\]'
        if re.search(markdown_task_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Markdown自动链接的文本
        markdown_autolink_pattern = r'<https?://\S+>|<\S+@\S+\.\S+>'
        if re.search(markdown_autolink_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown引用链接的文本
        markdown_reference_pattern = r'\[.*?\]:\s'
        if re.search(markdown_reference_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown注释的文本
        markdown_comment_pattern = r'<!--.*?-->'
        if re.search(markdown_comment_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown元数据的文本
        markdown_metadata_pattern = r'^---\s*$'
        if re.search(markdown_metadata_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown扩展语法的文本
        markdown_extension_pattern = r'^:::|^\|\|\|'
        if re.search(markdown_extension_pattern, text_stripped):
            return False
        
        # 过滤包含Emoji的文本
        emoji_pattern = r'[:][a-zA-Z0-9_+-]+[:]'
        if re.search(emoji_pattern, text_stripped):
            return False
        
        # 过滤包含特殊符号的文本
        special_symbol_pattern = r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]'
        if re.search(special_symbol_pattern, text_stripped):
            return False
        
        # 过滤包含标点符号的文本
        punctuation_pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]'
        if re.search(punctuation_pattern, text_stripped):
            return False
        
        # 过滤包含空格的文本
        if ' ' in text_stripped:
            return False
        
        # 过滤包含制表符的文本
        if '\t' in text_stripped:
            return False
        
        # 过滤包含换行符的文本
        if '\n' in text_stripped or '\r' in text_stripped:
            return False
        
        # 过滤包含回车符的文本
        if '\r' in text_stripped:
            return False
        
        # 过滤包含换页符的文本
        if '\f' in text_stripped:
            return False
        
        # 过滤包含垂直制表符的文本
        if '\v' in text_stripped:
            return False
        
        # 过滤包含其他空白字符的文本
        whitespace_pattern = r'\s'
        if re.search(whitespace_pattern, text_stripped):
            return False
        
        # 过滤包含不可见字符的文本
        invisible_pattern = r'[\x00-\x1F\x7F]'
        if re.search(invisible_pattern, text_stripped):
            return False
        
        # 过滤包含控制字符的文本
        control_pattern = r'[\x00-\x1F\x7F]'
        if re.search(control_pattern, text_stripped):
            return False
        
        # 过滤包含特殊字符的文本
        special_char_pattern = r'[\x00-\x1F\x7F!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]'
        if re.search(special_char_pattern, text_stripped):
            return False
        
        # 过滤包含非ASCII字符的文本（保留中文）
        non_ascii_pattern = r'[\x80-\xFF]'
        if re.search(non_ascii_pattern, text_stripped):
            # 检查是否包含中文
            chinese_pattern = r'[\u4e00-\u9fff]'
            if not re.search(chinese_pattern, text_stripped):
                return False
        
        # 过滤包含ASCII字符的文本（保留中文）
        ascii_pattern = r'[\x00-\x7F]'
        if re.search(ascii_pattern, text_stripped):
            # 检查是否包含中文
            chinese_pattern = r'[\u4e00-\u9fff]'
            if not re.search(chinese_pattern, text_stripped):
                return False
        
        # 过滤包含数字的文本
        digit_pattern = r'\d'
        if re.search(digit_pattern, text_stripped):
            return False
        
        # 过滤包含字母的文本
        alpha_pattern = r'[a-zA-Z]'
        if re.search(alpha_pattern, text_stripped):
            return False
        
        # 过滤包含混合字符的文本
        mixed_pattern = r'[\d][a-zA-Z]|[a-zA-Z][\d]'
        if re.search(mixed_pattern, text_stripped):
            return False
        
        # 过滤包含重复字符的文本
        repeat_pattern = r'(.)\1{2,}'
        if re.search(repeat_pattern, text_stripped):
            return False
        
        # 过滤包含连续空格的文本
        spaces_pattern = r'\s{2,}'
        if re.search(spaces_pattern, text_stripped):
            return False
        
        # 过滤包含首尾空格的文本
        if text_stripped != text_stripped.strip():
            return False
        
        # 过滤包含特殊格式的文本
        format_pattern = r'[*_~`#]'
        if re.search(format_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown格式的文本
        markdown_pattern = r'^#{1,6}\s|^-\s|^\d+\.\s|^>\s|^```'
        if re.search(markdown_pattern, text_stripped):
            return False
        
        # 过滤包含HTML格式的文本
        html_pattern = r'<[^>]+>'
        if re.search(html_pattern, text_stripped):
            return False
        
        # 过滤包含XML格式的文本
        xml_pattern = r'<[^>]+>'
        if re.search(xml_pattern, text_stripped):
            return False
        
        # 过滤包含JSON格式的文本
        json_pattern = r'\{\s*"|\[\s*"'
        if re.search(json_pattern, text_stripped):
            return False
        
        # 过滤包含YAML格式的文本
        yaml_pattern = r'^\w+:\s|^-\s'
        if re.search(yaml_pattern, text_stripped):
            return False
        
        # 过滤包含INI格式的文本
        ini_pattern = r'^\[\w+\]'
        if re.search(ini_pattern, text_stripped):
            return False
        
        # 过滤包含配置文件格式的文本
        config_pattern = r'^\w+\s*='
        if re.search(config_pattern, text_stripped):
            return False
        
        # 过滤包含日志格式的文本
        log_pattern = r'^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}'
        if re.search(log_pattern, text_stripped):
            return False
        
        # 过滤包含时间戳的文本
        timestamp_pattern = r'^\d{10}$|^\d{13}$'
        if re.search(timestamp_pattern, text_stripped):
            return False
        
        # 过滤包含UUID的文本
        uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        if re.search(uuid_pattern, text_stripped):
            return False
        
        # 过滤包含MAC地址的文本
        mac_pattern = r'^([0-9a-fA-F]{2}[:-]){5}([0-9a-fA-F]{2})$'
        if re.search(mac_pattern, text_stripped):
            return False
        
        # 过滤包含IP地址的文本
        ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        if re.search(ip_pattern, text_stripped):
            return False
        
        # 过滤包含IPv6地址的文本
        ipv6_pattern = r'^([0-9a-fA-F]{0,4}:){1,7}([0-9a-fA-F]{0,4})$'
        if re.search(ipv6_pattern, text_stripped):
            return False
        
        # 过滤包含URL路径的文本
        url_path_pattern = r'^/\S+'
        if re.search(url_path_pattern, text_stripped):
            return False
        
        # 过滤包含文件扩展名的文本
        file_ext_pattern = r'\.\w{1,5}$'
        if re.search(file_ext_pattern, text_stripped):
            return False
        
        # 过滤包含命令行参数的文本
        cli_pattern = r'^-\w+|^--\w+'
        if re.search(cli_pattern, text_stripped):
            return False
        
        # 过滤包含环境变量的文本
        env_pattern = r'^\$\w+'
        if re.search(env_pattern, text_stripped):
            return False
        
        # 过滤包含正则表达式的文本
        regex_pattern = r'^/.*?/'
        if re.search(regex_pattern, text_stripped):
            return False
        
        # 过滤包含代码注释的文本
        comment_pattern = r'^//|^#|^/\*|^\*/'
        if re.search(comment_pattern, text_stripped):
            return False
        
        # 过滤包含SQL语句的文本
        sql_pattern = r'^(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE|GRANT|REVOKE)\s'
        if re.search(sql_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Shell命令的文本
        shell_pattern = r'^\w+\s*\|\s*\w+|^\w+\s*;\s*\w+'
        if re.search(shell_pattern, text_stripped):
            return False
        
        # 过滤包含PowerShell命令的文本
        powershell_pattern = r'^\w+\s*\|\|\s*\w+|^\w+\s*&&\s*\w+'
        if re.search(powershell_pattern, text_stripped):
            return False
        
        # 过滤包含Python代码的文本
        python_pattern = r'^(import|from|def|class|if|elif|else|for|while|try|except|finally|with|as|pass|break|continue|return|yield|raise|assert|del|global|nonlocal|lambda|and|or|not|in|is|True|False|None)\s'
        if re.search(python_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含JavaScript代码的文本
        javascript_pattern = r'^(var|let|const|function|class|if|else|for|while|do|switch|case|default|try|catch|finally|with|return|break|continue|throw|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(javascript_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Java代码的文本
        java_pattern = r'^(public|private|protected|static|final|abstract|synchronized|native|transient|volatile|class|interface|enum|package|import|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(java_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含C/C++代码的文本
        cpp_pattern = r'^(public|private|protected|static|final|abstract|synchronized|native|transient|volatile|class|struct|union|enum|namespace|using|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(cpp_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含C#代码的文本
        csharp_pattern = r'^(public|private|protected|static|final|abstract|synchronized|native|transient|volatile|class|struct|union|enum|namespace|using|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(csharp_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Go代码的文本
        go_pattern = r'^(package|import|func|var|const|type|struct|interface|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(go_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Rust代码的文本
        rust_pattern = r'^(fn|let|const|static|struct|enum|trait|impl|mod|use|if|else|for|while|loop|match|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(rust_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含PHP代码的文本
        php_pattern = r'^(<?php|function|class|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(php_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Ruby代码的文本
        ruby_pattern = r'^(def|class|module|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(ruby_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Swift代码的文本
        swift_pattern = r'^(func|let|var|class|struct|enum|protocol|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(swift_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Kotlin代码的文本
        kotlin_pattern = r'^(fun|val|var|class|interface|enum|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(kotlin_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Dart代码的文本
        dart_pattern = r'^(void|int|double|String|bool|List|Map|Set|Function|class|interface|enum|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(dart_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含TypeScript代码的文本
        typescript_pattern = r'^(void|int|double|string|boolean|number|any|unknown|never|object|array|tuple|interface|type|class|enum|if|else|for|while|do|switch|case|default|try|catch|finally|return|break|continue|throw|throws|new|delete|typeof|instanceof|in|of|async|await|export|import|from|as|static|extends|super)\s'
        if re.search(typescript_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含HTML标签的文本
        html_tag_pattern = r'<[^>]+>'
        if re.search(html_tag_pattern, text_stripped):
            return False
        
        # 过滤包含XML标签的文本
        xml_tag_pattern = r'<[^>]+>'
        if re.search(xml_tag_pattern, text_stripped):
            return False
        
        # 过滤包含CSS选择器的文本
        css_selector_pattern = r'^\w+\s*{|^#\w+|^\.\w+'
        if re.search(css_selector_pattern, text_stripped):
            return False
        
        # 过滤包含正则表达式的文本
        regex_pattern = r'^/.*?/'
        if re.search(regex_pattern, text_stripped):
            return False
        
        # 过滤包含数学公式的文本
        math_pattern = r'^\$\$.*?\$\$|^\$.*?\$'
        if re.search(math_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown链接的文本
        markdown_link_pattern = r'\[.*?\]\(.*?\)'
        if re.search(markdown_link_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown图片的文本
        markdown_image_pattern = r'!\[.*?\]\(.*?\)'
        if re.search(markdown_image_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown标题的文本
        markdown_heading_pattern = r'^#{1,6}\s'
        if re.search(markdown_heading_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown列表的文本
        markdown_list_pattern = r'^-\s|^\d+\.\s'
        if re.search(markdown_list_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown引用的文本
        markdown_quote_pattern = r'^>\s'
        if re.search(markdown_quote_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown代码块的文本
        markdown_code_pattern = r'^```'
        if re.search(markdown_code_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown粗体的文本
        markdown_bold_pattern = r'\*\*.*?\*\*|__.*?__'
        if re.search(markdown_bold_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown斜体的文本
        markdown_italic_pattern = r'\*.*?\*|_.*?_' 
        if re.search(markdown_italic_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown删除线的文本
        markdown_strikethrough_pattern = r'~~.*?~~'
        if re.search(markdown_strikethrough_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown行内代码的文本
        markdown_inline_code_pattern = r'`.*?`'
        if re.search(markdown_inline_code_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown表格的文本
        markdown_table_pattern = r'^\|.*?\|'
        if re.search(markdown_table_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown分隔线的文本
        markdown_hr_pattern = r'^-{3,}|^\*{3,}|^_{3,}'
        if re.search(markdown_hr_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown脚注的文本
        markdown_footnote_pattern = r'\[\^.*?\]'
        if re.search(markdown_footnote_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown定义列表的文本
        markdown_definition_pattern = r'^\w+\s*:'
        if re.search(markdown_definition_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown任务列表的文本
        markdown_task_pattern = r'^-\s*\[ \]|^-\s*\[x\]'
        if re.search(markdown_task_pattern, text_stripped, re.IGNORECASE):
            return False
        
        # 过滤包含Markdown自动链接的文本
        markdown_autolink_pattern = r'<https?://\S+>|<\S+@\S+\.\S+>'
        if re.search(markdown_autolink_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown引用链接的文本
        markdown_reference_pattern = r'\[.*?\]:\s'
        if re.search(markdown_reference_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown注释的文本
        markdown_comment_pattern = r'<!--.*?-->'
        if re.search(markdown_comment_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown元数据的文本
        markdown_metadata_pattern = r'^---\s*$'
        if re.search(markdown_metadata_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown扩展语法的文本
        markdown_extension_pattern = r'^:::|^\|\|\|'
        if re.search(markdown_extension_pattern, text_stripped):
            return False
        
        # 过滤包含Emoji的文本
        emoji_pattern = r'[:][a-zA-Z0-9_+-]+[:]'
        if re.search(emoji_pattern, text_stripped):
            return False
        
        # 过滤包含特殊符号的文本
        special_symbol_pattern = r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]'
        if re.search(special_symbol_pattern, text_stripped):
            return False
        
        # 过滤包含标点符号的文本
        punctuation_pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]'
        if re.search(punctuation_pattern, text_stripped):
            return False
        
        # 过滤包含空格的文本
        if ' ' in text_stripped:
            return False
        
        # 过滤包含制表符的文本
        if '\t' in text_stripped:
            return False
        
        # 过滤包含换行符的文本
        if '\n' in text_stripped or '\r' in text_stripped:
            return False
        
        # 过滤包含回车符的文本
        if '\r' in text_stripped:
            return False
        
        # 过滤包含换页符的文本
        if '\f' in text_stripped:
            return False
        
        # 过滤包含垂直制表符的文本
        if '\v' in text_stripped:
            return False
        
        # 过滤包含其他空白字符的文本
        whitespace_pattern = r'\s'
        if re.search(whitespace_pattern, text_stripped):
            return False
        
        # 过滤包含不可见字符的文本
        invisible_pattern = r'[\x00-\x1F\x7F]'
        if re.search(invisible_pattern, text_stripped):
            return False
        
        # 过滤包含控制字符的文本
        control_pattern = r'[\x00-\x1F\x7F]'
        if re.search(control_pattern, text_stripped):
            return False
        
        # 过滤包含特殊字符的文本
        special_char_pattern = r'[\x00-\x1F\x7F!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]'
        if re.search(special_char_pattern, text_stripped):
            return False
        
        # 过滤包含非ASCII字符的文本（保留中文）
        non_ascii_pattern = r'[\x80-\xFF]'
        if re.search(non_ascii_pattern, text_stripped):
            # 检查是否包含中文
            chinese_pattern = r'[\u4e00-\u9fff]'
            if not re.search(chinese_pattern, text_stripped):
                return False
        
        # 过滤包含ASCII字符的文本（保留中文）
        ascii_pattern = r'[\x00-\x7F]'
        if re.search(ascii_pattern, text_stripped):
            # 检查是否包含中文
            chinese_pattern = r'[\u4e00-\u9fff]'
            if not re.search(chinese_pattern, text_stripped):
                return False
        
        # 过滤包含数字的文本
        digit_pattern = r'\d'
        if re.search(digit_pattern, text_stripped):
            return False
        
        # 过滤包含字母的文本
        alpha_pattern = r'[a-zA-Z]'
        if re.search(alpha_pattern, text_stripped):
            return False
        
        # 过滤包含混合字符的文本
        mixed_pattern = r'[\d][a-zA-Z]|[a-zA-Z][\d]'
        if re.search(mixed_pattern, text_stripped):
            return False
        
        # 过滤包含重复字符的文本
        repeat_pattern = r'(.)\1{2,}'
        if re.search(repeat_pattern, text_stripped):
            return False
        
        # 过滤包含连续空格的文本
        spaces_pattern = r'\s{2,}'
        if re.search(spaces_pattern, text_stripped):
            return False
        
        # 过滤包含首尾空格的文本
        if text_stripped != text_stripped.strip():
            return False
        
        # 过滤包含特殊格式的文本
        format_pattern = r'[*_~`#]'
        if re.search(format_pattern, text_stripped):
            return False
        
        # 过滤包含Markdown格式的文本
        markdown_pattern = r'^#{1,6}\s|^-\s|^\d+\.\s|^>\s|^```'
        if re.search(markdown_pattern, text_stripped):
            return False
        
        # 过滤包含HTML格式的文本
        html_pattern = r'<[^>]+>'
        if re.search(html_pattern, text_stripped):
            return False
        
        # 过滤包含XML格式的文本
        xml_pattern = r'<[^>]+>'
        if re.search(xml_pattern, text_stripped):
            return False
        
        # 过滤包含JSON格式的文本
        json_pattern = r'\{\s*"|\[\s*"'
        if re.search(json_pattern, text_stripped):
            return False
        
        # 过滤包含YAML格式的文本
        yaml_pattern = r'^\w+:\s|^-\s'
        if re.search(yaml_pattern, text_stripped):
            return False
        
        # 如果通过所有过滤，则认为是角色名称
        return True
    
    def load_characters_from_file(self, file_path):
        """
        从文件加载角色名称
        
        Args:
            file_path: 文件路径
            
        Returns:
            角色名称列表
        """
        characters = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if self.is_character_name(line):
                        characters.append(line)
            
            logger.info(f"从 {file_path} 加载了 {len(characters)} 个角色名称")
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
        
        return characters
    
    def load_characters_from_directory(self, directory_path):
        """
        从目录加载所有角色文件
        
        Args:
            directory_path: 目录路径
            
        Returns:
            角色名称字典 {文件名: [角色名称]}
        """
        characters_dict = {}
        
        try:
            for file_name in os.listdir(directory_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(directory_path, file_name)
                    characters = self.load_characters_from_file(file_path)
                    if characters:
                        characters_dict[file_name] = characters
            
            total_characters = sum(len(chars) for chars in characters_dict.values())
            logger.info(f"从 {directory_path} 加载了 {total_characters} 个角色名称")
        except Exception as e:
            logger.error(f"加载目录失败 {directory_path}: {e}")
        
        return characters_dict
    
    def validate_character_names(self, characters):
        """
        验证角色名称列表
        
        Args:
            characters: 角色名称列表
            
        Returns:
            验证后的角色名称列表
        """
        validated_characters = []
        
        for character in characters:
            if self.is_character_name(character):
                validated_characters.append(character)
        
        logger.info(f"验证了 {len(characters)} 个角色名称，通过 {len(validated_characters)} 个")
        return validated_characters
    
    def save_characters(self, characters, output_file):
        """
        保存角色名称到文件
        
        Args:
            characters: 角色名称列表
            output_file: 输出文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for character in characters:
                    f.write(character + '\n')
            
            logger.info(f"保存了 {len(characters)} 个角色名称到 {output_file}")
        except Exception as e:
            logger.error(f"保存文件失败 {output_file}: {e}")


def main():
    """
    主函数
    """
    collector = CharacterNameCollector()
    
    # 示例：从目录加载角色文件
    characters_dir = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/auto_spider_img/characters'
    characters_dict = collector.load_characters_from_directory(characters_dir)
    
    # 打印结果
    for file_name, characters in characters_dict.items():
        print(f"文件: {file_name}")
        print(f"角色数量: {len(characters)}")
        print(f"前5个角色: {characters[:5]}")
        print()


if __name__ == '__main__':
    main()
