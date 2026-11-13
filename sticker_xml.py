import random
import xml.etree.ElementTree as ET


# 框架的 add_message_ids() 是按「原始 XML 中第 i 个 <msg>」回填「第 i 条发送结果」的
# message_id。任何在发送前增删、拆分、重排消息块的做法都会让这个对应关系错位，
# 于是回写的 message_id 贴到了错误的消息上，之后模型再引用/撤回/贴表情就会指错目标。
# 这里改为直接改写 XML 本身，让 <msg> 序列与实际发送序列严格一一对应。
_STICKER_MSG = "msg"
_STICKER_TAG = "sticker"
_STICKER_REPLY = "reply"
_STICKER_TEXT = "text"
# 消息全部被过滤掉时用它表示「这一轮不发送任何消息」，与提示词里 <msg/> 的语义一致
_STICKER_EMPTY_TURN = ""


def _sticker_is_blank_text(node) -> bool:
    """<text></text>、<text>   </text> 解析后不会产生任何消息元素。"""
    return node.tag == _STICKER_TEXT and not (node.text or "").strip()


def _sticker_effective_children(msg) -> list:
    """剔除解析后不产生元素的空白 <text>，得到真正会变成消息元素的子节点。"""
    return [n for n in msg if not _sticker_is_blank_text(n)]


def _sticker_produces_element(node, sticker_enabled, valid_sticker_ids) -> bool:
    """该子节点解析后是否真的会变成一个消息元素。

    <text> 空白时不产出；<sticker> 在标签未注册、或 id 不在表情库里时同样不产出
    （内置 StickerTag 对未知 id 返回空列表）。其余标签一律按会产出处理。
    """
    if node.tag == _STICKER_TEXT:
        return bool((node.text or "").strip())
    if node.tag == _STICKER_TAG:
        if not sticker_enabled:
            return False
        if valid_sticker_ids is None:
            return True
        return (node.text or "").strip() in valid_sticker_ids
    return True


def _sticker_live_children(msg, sticker_enabled, valid_sticker_ids) -> list:
    """真正会产出消息元素的子节点（用于判断这条消息发出去还有没有东西）。"""
    return [n for n in msg
            if _sticker_produces_element(n, sticker_enabled, valid_sticker_ids)]


def _sticker_detach(parent, child) -> None:
    """从 parent 摘下 child，并把它的 tail 并回前一个兄弟，避免文本/空白错位。"""
    children = list(parent)
    idx = children.index(child)
    tail = child.tail or ""
    parent.remove(child)
    if idx > 0:
        prev = children[idx - 1]
        prev.tail = (prev.tail or "") + tail
    else:
        parent.text = (parent.text or "") + tail


def _sticker_detach_from_root(root, node) -> int:
    """从 root 摘下 node，返回它原来的下标。"""
    children = list(root)
    idx = children.index(node)
    tail = node.tail or ""
    root.remove(node)
    if idx > 0:
        children[idx - 1].tail = (children[idx - 1].tail or "") + tail
    else:
        root.text = (root.text or "") + tail
    return idx


def _sticker_make_msg(node):
    """把一个表情元素包装成独立的消息块。"""
    msg = ET.Element(_STICKER_MSG)
    msg.append(node)
    msg.tail = "\n"
    return msg


def _sticker_is_sticker_only(node) -> bool:
    return node.tag == _STICKER_MSG and [n.tag for n in node] == [_STICKER_TAG]


def _sticker_order(root) -> list:
    """消息的顺序签名，用于判断一次重排是否真的改变了位置。"""
    return [(n.tag, (n.text or "").strip() or (n[0].text or "").strip() if len(n) else "")
            for n in root]


def _sticker_signature(root) -> list:
    """整棵树的可比较签名，用于改写后的语义自检。"""
    return [
        (node.tag, tuple(sorted(node.attrib.items())),
         tuple((c.tag, tuple(sorted(c.attrib.items())), (c.text or "").strip()) for c in node))
        for node in root
    ]


def _sticker_shuffle(root, rng) -> None:
    """在「根标签分隔出的片段」内重排表情消息，绝不跨过根标签。

    根标签（如 <mute/>）代表一个动作，表情不能改变它的相对执行位置。
    """
    segments = []
    segment = []
    for node in list(root):
        if node.tag == _STICKER_MSG:
            segment.append(node)
        else:
            if segment:
                segments.append(segment)
                segment = []
    if segment:
        segments.append(segment)

    for seg in segments:
        stickers = [n for n in seg if _sticker_is_sticker_only(n)]
        if not stickers:
            continue
        others = [n for n in seg if not _sticker_is_sticker_only(n)]
        anchor = list(root).index(seg[0])
        for node in seg:
            root.remove(node)
        pool = others[:]
        for sc in stickers:
            pool.insert(rng.randint(0, len(pool)), sc)
        for offset, node in enumerate(pool):
            root.insert(anchor + offset, node)


def rewrite_sticker_xml(xml_data, probability, random_position, rng=None,
                         sticker_enabled=True, valid_sticker_ids=None):
    """把 <sticker> 拆成独立消息块、丢弃空引用消息，并按需重排位置。

    sticker_enabled: sticker 标签本轮是否已注册。未注册时该标签解析不出任何元素，
        不能拆（否则会凭空多出一条空消息，反而破坏 message_id 对齐）。
    valid_sticker_ids: 当前可用的 sticker id 集合。标签遇到未知 id 会返回空列表，
        先把这些节点摘掉，避免同一问题。

    返回改写后的 XML；无需改写或无法安全改写时返回 None，调用方必须保持原文。
    """
    if not isinstance(xml_data, str) or not xml_data or _STICKER_MSG not in xml_data:
        return None
    try:
        root = ET.fromstring(f"<root>{xml_data}</root>")
    except ET.ParseError:
        # 交给框架自己的 XML 修复流程，这里绝不碰原文
        return None

    rng = rng or random
    changed = False
    # 内容不需要改动、但仍要参与重排时置位（否则 random_position 会被跳过）
    shuffle_only = False

    for msg in list(root):
        if msg.tag != _STICKER_MSG or not list(msg):
            # 空的 <msg/> 是「不发消息」的语义，留到最后统一处理
            continue

        # 解析不出元素的表情先摘掉，否则它会把消息撑成「还有内容」
        if valid_sticker_ids is not None:
            for node in [n for n in msg if n.tag == _STICKER_TAG]:
                if (node.text or "").strip() not in valid_sticker_ids:
                    _sticker_detach(msg, node)
                    changed = True

        stickers = [n for n in msg if n.tag == _STICKER_TAG] if sticker_enabled else []
        removed = False

        if stickers:
            # 按概率决定哪些表情保留
            kept = [n for n in stickers if rng.random() < probability]
            rest = [n for n in _sticker_effective_children(msg) if n.tag != _STICKER_TAG]

            # 单表情、被保留、且没有其他内容：内容无需改动，
            # 但 random_position 仍要生效——它本来就是独立一条，
            # 需要在后面的重排里参与位置随机，不能在这里提前退出。
            if len(stickers) == 1 and kept and not rest:
                shuffle_only = True
            else:
                changed = True
                # 原链中移除所有表情（无论是否保留）
                for node in stickers:
                    _sticker_detach(msg, node)

                if not rest:
                    # 纯表情：原消息不再产生任何元素，保留的表情各自独立成条
                    insert_at = _sticker_detach_from_root(root, msg)
                    removed = True
                else:
                    # 还有正文：正文留在原消息，保留的表情紧跟其后独立成条
                    insert_at = list(root).index(msg) + 1
                    # 去掉表情后只剩一个 <reply> 的「空引用」整条丢弃
                    if len(rest) == 1 and rest[0].tag == _STICKER_REPLY:
                        insert_at = _sticker_detach_from_root(root, msg)
                        removed = True

                for node in kept:
                    root.insert(insert_at, _sticker_make_msg(node))
                    insert_at += 1

        # 只包含一个 <reply> 的「空引用」消息块整条丢弃（与原有行为一致）。
        # 判定用「真正会产出元素」的子节点：未注册或 id 无效的 <sticker> 不产出任何
        # 东西，不能让它把空引用撑成「有内容」，否则会发出一个只有引用的空气泡。
        if not removed:
            rest = _sticker_live_children(msg, sticker_enabled, valid_sticker_ids)
            if len(rest) == 1 and rest[0].tag == _STICKER_REPLY:
                _sticker_detach_from_root(root, msg)
                changed = True

    if not changed and not (random_position and shuffle_only):
        return None

    # 清理解析后不会产生任何消息元素的消息块（空 <msg/>、只剩空白 <text> 的、
    # 表情被摘光又没有其他内容的）。它们若不删掉，XML 的 <msg> 数就会多于实际发送
    # 条数，message_id 依然会错位。
    for msg in list(root):
        if msg.tag == _STICKER_MSG and not _sticker_live_children(
                msg, sticker_enabled, valid_sticker_ids):
            _sticker_detach_from_root(root, msg)

    if random_position:
        before = _sticker_order(root)
        _sticker_shuffle(root, rng)
        # 纯重排路径下如果顺序没变（只有一个候选位置等），保持原文不动
        if not changed and _sticker_order(root) == before:
            return None

    # 消息全被过滤掉、且没有根标签（动作）要执行时，返回空响应。
    # 保持空字符串而不是 <msg/>：既不会发送任何气泡，也不会被其他插件的
    # 「空消息即停止」判定误认为本轮是主动静默结束。
    if not [n for n in root if n.tag == _STICKER_MSG]:
        leftovers = [n for n in root if n.tag != _STICKER_MSG]
        if not leftovers:
            return _STICKER_EMPTY_TURN

    expected = _sticker_signature(root)
    body = ET.tostring(root, encoding="unicode", method="xml")[6:-7]

    # 序列化保真自检：重新解析后语义必须与内存中的树一致，否则放弃改写
    try:
        check = ET.fromstring(f"<root>{body}</root>")
    except ET.ParseError:
        return None
    if _sticker_signature(check) != expected:
        return None

    return body
