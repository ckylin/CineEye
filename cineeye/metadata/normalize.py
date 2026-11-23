"""片名归一化与模糊匹配：为「一部电影多个名字」提供搜索能力。

用途：把任意写法的片名（繁体/简体、大小写、标点空格差异）转成统一形式，
    让同一部电影的多个译名能被检索召回；similarity() 在归一化基础上做
    序列相似度打分，tokens() 提供简单的词级召回。
被谁使用：metadata/store.py 的搜索与演员 id 生成（_ensure_person）依赖本模块。
依赖：仅标准库（unicodedata / re / difflib），零第三方依赖。
运维要点（改这里之前必须知道的事）：
    1. _T2S 是手工维护的高频繁→简映射表，只覆盖电影片名常用字；如果搜索
       召回出现繁体漏配，优先怀疑是映射表缺字，往表里加即可（不要引入
       opencc 重依赖，除非确实需要全量字表）；
    2. 归一化逻辑一旦改动，persons 表里已落盘的 person_id（MD5(归一化名)）
       会与新算法不一致，导致演员重复建行——改之前先评估存量数据；
    3. normalize 的三个开关（case/punct/t2s）由 store 的 metadata.aliases.*
       配置控制，改默认值会影响搜索行为的兼容性。
"""

from __future__ import annotations

import re
import unicodedata

# 常用繁体 → 简体映射（覆盖电影片名高频字，避免引入 opencc 等重依赖）
_T2S = {
    "愛": "爱", "無": "无", "為": "为", "與": "与", "時": "时", "間": "间", "間": "间",
    "電": "电", "視": "视", "邊": "边", "緣": "缘", "漢": "汉", "資": "资", "訊": "讯",
    "讓": "让", "認": "认", "識": "识", "夢": "梦", "國": "国", "學": "学", "國": "国",
    "導": "导", "演": "演", "員": "员", "劇": "剧", "場": "场", "景": "景", "風": "风",
    "雲": "云", "龍": "龙", "鳳": "凤", "後": "后", "裡": "里", "幾": "几", "個": "个",
    "們": "们", "來": "来", "去": "去", "過": "过", "還": "还", "這": "这", "那": "那",
    "說": "说", "話": "话", "語": "语", "聲": "声", "音": "音", "樂": "乐", "畫": "画",
    "面": "面", "對": "对", "開": "开", "關": "关", "發": "发", "現": "现", "實": "实",
    "體": "体", "會": "会", "議": "议", "決": "决", "策": "策", "劃": "划", "計": "计",
    "節": "节", "目": "目", "錄": "录", "張": "张", "東": "东", "方": "方",
    "車": "车", "馬": "马", "鳥": "鸟", "魚": "鱼", "花": "花", "草": "草", "樹": "树",
    "葉": "叶", "果": "果", "紅": "红", "藍": "蓝", "綠": "绿", "黃": "黄",
    "黑": "黑", "白": "白", "級": "级", "線": "线", "橫": "横", "縱": "纵", "長": "长",
    "短": "短", "高": "高", "低": "低", "大": "大", "小": "小", "多": "多", "少": "少",
    "新": "新", "舊": "旧", "好": "好", "壞": "坏", "真": "真", "假": "假", "無": "无",
}
_TRANS_TABLE = str.maketrans(_T2S)

_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize(text: str, *, case: bool = True, punct: bool = True, t2s: bool = True) -> str:
    """把片名转成可比较的形式：NFKC → 繁转简 → 小写 → 去标点空白。

    归一化是检索召回的关键：用户搜「那些年，我们一起追的女孩」时，
    库里的「那些年，我們一起追的女孩。」必须先归一化成相同字符串才能精确命中；
    其余写法靠 similarity() 的模糊打分兜底。

    Args:
        text: 原始片名。
        case: 是否转小写（对西文片名有效）。
        punct: 是否去除所有标点、空白与下划线。
        t2s: 是否做繁体→简体映射。

    Returns:
        归一化后的字符串（可能为空串）。
    """
    # NFKC 先做 Unicode 兼容分解：全角字母/数字、特殊符号统一成常规形式
    s = unicodedata.normalize("NFKC", str(text))
    if t2s:
        s = s.translate(_TRANS_TABLE)  # 繁体片名（港台译名常见）映射为简体，提升中文召回
    if case:
        s = s.lower()  # 英文片名大小写统一
    if punct:
        s = _PUNCT.sub("", s)  # 去掉《》「」- 空格等标点，让「复仇者联盟4」与「复仇者联盟 4」相等
    return s


def similarity(a: str, b: str, *, case: bool = True, punct: bool = True, t2s: bool = True) -> float:
    """归一化后的序列相似度 0~1（difflib，零依赖）。

    用于精确归一化匹配失败后的模糊兜底：处理「变形金刚4：绝迹重生」vs
    「变形金刚4 绝迹重生」这类残缺/多余字的情况。

    Args:
        a: 片名 A。
        b: 片名 B。
        case/punct/t2s: 与 normalize 相同的归一化开关。

    Returns:
        0~1 的相似度；任一输入归一化后为空返回 0.0；完全相等返回 1.0。
    """
    from difflib import SequenceMatcher

    na, nb = normalize(a, case=case, punct=punct, t2s=t2s), normalize(b, case=case, punct=punct, t2s=t2s)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return float(SequenceMatcher(None, na, nb).ratio())


def tokens(text: str, *, case: bool = True, punct: bool = True, t2s: bool = True) -> set[str]:
    """简单中文/西文切分后的词集合，用于召回。

    中文按 1~4 字滑窗切分（无分词依赖的粗糙方案），西文/数字按连续串切分。
    当前主要作为候选扩展接口，store.search 走的是整体相似度。

    Args:
        text: 原始文本。
        case/punct/t2s: 归一化开关（punct 只用于 normalize，切分时保留空白结构）。

    Returns:
        词集合 set[str]。
    """
    s = normalize(text, case=case, punct=False, t2s=t2s)
    parts = re.findall(r"[\u4e00-\u9fff]{1,4}|[a-z0-9]+", s.lower() if case else s)
    return {p for p in parts if p}
