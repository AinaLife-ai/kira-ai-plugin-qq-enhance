import asyncio
import json
import random
import re
import xml.etree.ElementTree as ET
from core.plugin import BasePlugin, PluginContext, logger, on, Priority, register
from core.chat.message_utils import KiraMessageBatchEvent, KiraMessageEvent
from core.chat.message_elements import Text, Sticker, Reply
from core.chat import MessageChain, Session
from core.tag import tag_registry
from core.provider import LLMRequest, LLMResponse
from core.utils.tool_utils import BaseTool


# ==================== 表情控制（XML 层改写） ====================
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


def _sticker_rewrite_xml(xml_data, probability, random_position, rng=None,
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


# ==================== 表情控制（XML 层改写）结束 ====================

# ==================== 原有工具类（贴表情、点赞、撤回、禁言） ====================
class SetEmojiTool(BaseTool):
    name = "set_qq_emoji"
    description = "给QQ消息贴表情"
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "QQ消息ID"},
            "emoji_id": {"type": "string", "description": "表情ID，和<emoji>标签的表情ID相同"}
        },
        "required": ["message_id", "emoji_id"]
    }

    def __init__(self, ctx: PluginContext):
        super().__init__(ctx=ctx)

    async def execute(self, event: KiraMessageBatchEvent, *args, message_id: str, emoji_id: str, **kwargs) -> str:
        ada_name = event.session.adapter_name
        ada = self.ctx.adapter_mgr.get_adapter(ada_name)
        client = ada.get_client()
        params = {
            "message_id": message_id,
            "emoji_id": emoji_id
        }
        res = await client.send_action("set_msg_emoji_like", params)
        return res


class SendQQLikesTool(BaseTool):
    name = "send_qq_likes"
    description = "给QQ用户资料卡点赞"
    parameters = {
        "type": "object",
        "properties": {
            "qq": {"type": "string", "description": "QQ账号"},
            "times": {"type": "integer", "description": "点赞次数，默认为最大可点赞数，除非用户要求，否则无需改动"}
        },
        "required": ["qq"]
    }

    def __init__(self, ctx: PluginContext):
        super().__init__(ctx=ctx)

    async def execute(self, event: KiraMessageBatchEvent, *args, qq: str, times: int = 50, **kwargs) -> str:
        ada_name = event.session.adapter_name
        ada = self.ctx.adapter_mgr.get_adapter(ada_name)
        client = ada.get_client()
        if not client:
            return "点赞失败，未找到当前QQ适配器客户端"

        try:
            times = int(times)
        except (TypeError, ValueError):
            return "点赞失败：点赞次数参数无效"
        if times < 1:
            return "点赞失败：点赞次数必须大于0"

        chunks = [10] * (times // 10) + ([times % 10] if times % 10 else [])
        state = {"likes_count": 0, "fail_msg": ""}
        try:
            await asyncio.wait_for(self._do_send_likes(client, qq, chunks, state), timeout=15)
        except asyncio.TimeoutError:
            if state["likes_count"]:
                # 超时前点出的赞已生效，同样按部分成功上报
                return f"点赞成功，共点了 {state['likes_count']} 个赞（请求超时中断，未能点满 {times} 个）"
            return "点赞超时"
        if state["fail_msg"]:
            if state["likes_count"]:
                # 部分成功：已点出的赞真实生效（常见于达到每日可赞上限被截断），
                # 必须以成功口径上报实际数量，否则LLM会误判为一个赞都没点上
                return f"点赞成功，共点了 {state['likes_count']} 个赞（未能点满 {times} 个：{state['fail_msg']}）"
            return f"点赞失败：{state['fail_msg']}"
        return f"点赞成功，共点了 {state['likes_count']} 个赞"

    @staticmethod
    async def _do_send_likes(client, qq: str, chunks: list[int], state: dict) -> None:
        for chunk in chunks:
            resp = await client.send_action("send_like", {"user_id": qq, "times": chunk})
            ok = isinstance(resp, dict) and (resp.get("status") == "ok" or resp.get("retcode") == 0)
            if ok:
                state["likes_count"] += chunk
                await asyncio.sleep(0.1)
                continue
            raw = resp.get("message") or resp.get("wording") if isinstance(resp, dict) else str(resp)
            # NapCat 抛错为"点赞失败 xxx"，去掉前缀避免拼出"点赞失败：点赞失败 xxx"
            state["fail_msg"] = re.sub(r"^点赞失败[:：\s]*", "", str(raw or "未知错误"))
            # 每日可赞上限用尽后继续请求必然同样失败，直接终止
            return


class DeleteMsgTool(BaseTool):
    name = "delete_qq_msg"
    description = "撤回QQ消息"
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "要撤回的QQ消息ID"},
        },
        "required": ["message_id"]
    }

    def __init__(self, ctx: PluginContext):
        super().__init__(ctx=ctx)

    async def execute(self, event: KiraMessageBatchEvent, *args, message_id: str, **kwargs) -> str:
        ada_name = event.session.adapter_name
        ada = self.ctx.adapter_mgr.get_adapter(ada_name)
        client = ada.get_client()
        params = {
            "message_id": message_id,
        }
        res = await client.send_action("delete_msg", params)
        return res


class GroupBanTool(BaseTool):
    name = "set_qq_group_ban"
    description = "禁言QQ群中的成员"
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string", "description": "要禁言的成员的QQ号"},
            "duration": {"type": "integer", "description": "禁言时长（秒），默认600秒（10分钟），设为0秒即为解除禁言", "default": 600},
        },
        "required": ["user_id", "duration"]
    }

    def __init__(self, ctx: PluginContext):
        super().__init__(ctx=ctx)

    async def execute(self, event: KiraMessageBatchEvent, *args, user_id: str, duration: int = 600, **kwargs) -> str:
        ada_name = event.session.adapter_name
        ada = self.ctx.adapter_mgr.get_adapter(ada_name)
        client = ada.get_client()
        params = {
            "group_id": event.session.session_id,
            "user_id": user_id,
            "duration": duration
        }
        res = await client.send_action("set_group_ban", params)
        return res


class SendMusicCardTool(BaseTool):
    name = "send_music_card"
    description = "发送音乐卡片到当前QQ聊天"
    parameters = {
        "type": "object",
        "properties": {
            "platform": {"type": "string", "description": "音乐平台类型，qq为QQ音乐，163为网易云音乐", "enum": ["qq", "163", "kugou", "migu", "kuwo"]},
            "music_id": {"type": "string", "description": "音乐ID，具体平台的ID格式不同，请根据平台要求提供正确的ID"}
        },
        "required": ["platform", "music_id"]
    }

    def __init__(self, ctx: PluginContext):
        super().__init__(ctx=ctx)

    async def execute(self, event: KiraMessageBatchEvent, *args, platform: str, music_id: str, **kwargs) -> str:
        ada_name = event.session.adapter_name
        ada = self.ctx.adapter_mgr.get_adapter(ada_name)
        client = ada.get_client()

        is_group = event.is_group_message()

        params = {
            "group_id" if is_group else "user_id": event.session.session_id,
            "message": [
                {
                    "type": "music",
                    "data": {
                        "type": platform,
                        "id": music_id
                    }
                }
            ]
        }
        res = await client.send_action("send_msg", params)
        return res


# ==================== 主插件类（合并所有功能） ====================
class QQEnhancePlugin(BasePlugin):
    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)

        # ----- 原有配置 -----
        self.emoji_react_enabled = self.plugin_cfg.get("emoji_react_enabled", True)
        self.send_likes_enabled = self.plugin_cfg.get("send_likes_enabled", False)
        self.delete_msg_enabled = self.plugin_cfg.get("delete_msg_enabled", True)
        self.group_ban_enabled = self.plugin_cfg.get("group_ban_enabled", True)
        self.qq_enhance_prompt = self.plugin_cfg.get("qq_enhance_prompt", "")
        self.perceive_group_ban = self.plugin_cfg.get("perceive_group_ban", True)
        self.perceive_group_increase = self.plugin_cfg.get("perceive_group_increase", True)
        self.music_card_enabled = self.plugin_cfg.get("music_card_enabled", False)

        # ----- Sticker Control 配置 -----
        self.sticker_control_enabled = self.plugin_cfg.get("sticker_control_enabled", False)
        self.sticker_probability = float(self.plugin_cfg.get("sticker_probability", 1.0))
        self.random_position = bool(self.plugin_cfg.get("random_position", True))

        # ----- Typing Indicator 配置 -----
        self.typing_indicator_enabled = self.plugin_cfg.get("typing_indicator_enabled", True)
        self.typing_delay_seconds = float(self.plugin_cfg.get("typing_delay_seconds", 2.0))
        self.typing_interval_seconds = float(self.plugin_cfg.get("typing_interval_seconds", 2.0))
        self.typing_max_seconds = float(self.plugin_cfg.get("typing_max_seconds", 60.0))

        # 用于 Typing Indicator 的状态管理
        self._delay_tasks = {}
        self._loop_tasks = {}
        self._max_tasks = {}
        self._typing_running = {}

    async def initialize(self):
        logger.info(f"QQEnhancePlugin initialized")

    async def terminate(self):
        # 清理 Typing Indicator 任务
        for task in self._delay_tasks.values():
            if not task.done():
                task.cancel()
        for task in self._loop_tasks.values():
            if not task.done():
                task.cancel()
        for task in self._max_tasks.values():
            if not task.done():
                task.cancel()
        self._delay_tasks.clear()
        self._loop_tasks.clear()
        self._max_tasks.clear()
        self._typing_running.clear()

    async def _get_qq_user_name(self, event: KiraMessageEvent, user_id, group_id=None) -> str:
        """Resolve a QQ user's display name, preferring the group card over the account nickname.

        Returns an empty string when the name cannot be resolved.
        """
        if not user_id:
            return ""
        try:
            ada = self.ctx.adapter_mgr.get_adapter(event.adapter.name)
            client = ada.get_client() if ada else None
            if not client:
                return ""
            if group_id:
                resp = await client.send_action(
                    "get_group_member_info",
                    {"group_id": group_id, "user_id": user_id, "no_cache": False}
                )
                data = (resp or {}).get("data") or {}
                name = data.get("card") or data.get("nickname")
                if name:
                    return str(name)
            resp = await client.get_user_info(user_id=user_id)
            return str(((resp or {}).get("data") or {}).get("nickname") or "")
        except Exception as e:
            logger.warning(f"Failed to resolve QQ user name for {user_id}: {e}")
            return ""

    @on.im_message(priority=Priority.HIGH + 1)
    async def perceive_notice(self, event: KiraMessageEvent):
        """感知QQ特有的notice事件，如群禁言、成员增加等，并将相关信息附加到消息链中，供后续使用"""
        if event.adapter.platform != "QQ":
            return
        if not event.is_notice:
            return

        msg = event.message.raw_message
        if not isinstance(msg, dict):
            return
        message_chain = event.message.chain

        notice_type = msg.get("notice_type")
        sub_type = msg.get("sub_type")
        self_id = msg.get("self_id")
        user_id = msg.get("user_id")
        target_id = msg.get("target_id")
        group_id = msg.get("group_id")

        if notice_type == "group_ban" and self_id == user_id and self.perceive_group_ban:
            event.message.is_mentioned = True

            ban_duration = msg.get("duration")
            ban_operator_id = msg.get("operator_id")
            ban_group_id = msg.get("group_id")
            # Resolve the operator's nickname so the notice is not id-only
            ban_operator_name = await self._get_qq_user_name(event, ban_operator_id, ban_group_id)
            operator_text = (f"用户{ban_operator_id}({ban_operator_name})"
                             if ban_operator_name else f"用户{ban_operator_id}")
            if sub_type == "ban":
                message_chain.text(f"[System {operator_text}禁言了你{ban_duration}秒]")

            elif sub_type == "lift_ban":  # 人为解除禁言
                # ban_duration 永远是0，invalid
                message_chain.text(f"[System 你之前被禁言了，{operator_text}解除了你的禁言]")
            else:
                return

        # --------- 新成员进群 ---------
        elif notice_type == "group_increase" and self.perceive_group_increase:
            # and msg["sub_type"] == "approve"
            if not group_id:
                return

            event.message.is_mentioned = True

            message_chain.text(f"[System 用户{user_id}加入了群聊]")
        else:
            pass

    # ---------- 注入工具 ----------
    @on.llm_request()
    async def inject_qq_enhance_tools(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        """根据配置注入QQ相关工具，并将工具说明添加到 system prompt 中"""
        platform = event.adapter.platform
        if not platform == "QQ":
            return

        if self.emoji_react_enabled and event.is_group_message():
            req.tool_set.add(SetEmojiTool(ctx=self.ctx))
            # 注入工具说明到 system prompt
            for p in req.system_prompt:
                if p.name == "tools":
                    p.content += f"\n{self.qq_enhance_prompt}"
                    break

        if self.send_likes_enabled:
            req.tool_set.add(SendQQLikesTool(ctx=self.ctx))

        if self.delete_msg_enabled:
            req.tool_set.add(DeleteMsgTool(ctx=self.ctx))

        if self.group_ban_enabled and event.is_group_message():
            req.tool_set.add(GroupBanTool(ctx=self.ctx))

        if self.music_card_enabled:
            req.tool_set.add(SendMusicCardTool(ctx=self.ctx))

    # ---------- Sticker Control 功能 ----------
    # 说明：表情的拆分/丢弃/重排必须在 XML 阶段完成。
    # 框架的 add_message_ids() 按「原始 XML 中第 i 个 <msg>」回填「第 i 条发送结果」的
    # message_id；若在 after_xml_parse 里增删或重排消息块，这个一一对应关系就被破坏，
    # 回写的 message_id 会贴到错误的消息上，之后引用/撤回/贴表情都会指向错误目标。
    # 因此这里改为在 llm_response 阶段直接改写 XML，让 <msg> 序列与实际发送序列保持一致。
    @on.llm_request(priority=Priority.LOW)
    async def remember_sticker_tag(self, event: KiraMessageBatchEvent, _req, tag_set):
        # sticker 标签未注册时（内置表情插件被关掉等）它解析不出任何元素，
        # 这时不能把 <sticker> 拆成独立消息，否则会凭空多出一条空消息。
        # 注意时机：框架是在所有 llm_request 处理器跑完之后，才把插件用
        # @register.tag 声明的标签补进 tag_set，所以这里还要看一眼全局注册表，
        # 否则装饰器注册的 sticker 标签会被误判为未注册，导致功能静默失效。
        ready = "sticker" in tag_set or "sticker" in tag_registry
        setattr(event, "_sticker_tag_ready", ready)

    @on.llm_response(priority=Priority.LOW)
    async def process_stickers(self, event: KiraMessageBatchEvent, resp: LLMResponse):
        if not self.sticker_control_enabled:
            return
        if resp.status == "error":
            return
        rewritten = _sticker_rewrite_xml(
            resp.text_response,
            self.sticker_probability,
            self.random_position,
            sticker_enabled=getattr(event, "_sticker_tag_ready", False),
            valid_sticker_ids=self._sticker_ids(),
        )
        if rewritten is None:
            # 没有表情、无需过滤，或 XML 无法安全改写：保持原文不动
            return
        logger.debug(f"Sticker 处理完成，改写后的消息块数量: {rewritten.count('<msg')}")
        resp.text_response = rewritten

    def _sticker_ids(self):
        """当前可用的 sticker id 集合；拿不到时返回 None（不做校验）。"""
        try:
            sticker_mgr = getattr(self.ctx, "sticker_mgr", None)
            if sticker_mgr is None:
                return None
            return set(sticker_mgr.sticker_dict)
        except Exception:
            return None
    # ---------- Typing Indicator 功能 ----------
    async def _send_typing(self, session: Session, log: bool = True):
        if not self.typing_indicator_enabled:
            return

        # 群聊不发送
        if session.session_type == "gm":
            return

        adapter = self.ctx.adapter_mgr.get_adapter(session.adapter_name)
        if not adapter:
            logger.error(f"Adapter '{session.adapter_name}' not found")
            return

        client = adapter.get_client()
        if not client:
            logger.error("Adapter client not available")
            return

        params = {"user_id": int(session.session_id), "event_type": 1}
        action = "set_input_status"

        if hasattr(client, 'send_action') and callable(client.send_action):
            try:
                await client.send_action(action, params)
                if log:
                    logger.debug(f"Typing sent to {session.sid}")
                return
            except Exception as e:
                logger.debug(f"send_action failed: {e}")

        # 尝试 WebSocket 发送
        ws = getattr(client, 'ws', None)
        if ws and hasattr(ws, 'send'):
            payload = json.dumps({"action": action, "params": params})
            try:
                await ws.send(payload)
                if log:
                    logger.debug(f"Typing sent via WebSocket to {session.sid}")
                return
            except Exception as e:
                logger.debug(f"WebSocket send failed: {e}")

        # 尝试其他可能属性
        for attr in ['_ws', '_client', 'websocket']:
            ws_attr = getattr(client, attr, None)
            if ws_attr and hasattr(ws_attr, 'send'):
                payload = json.dumps({"action": action, "params": params})
                try:
                    await ws_attr.send(payload)
                    if log:
                        logger.debug(f"Typing sent via {attr} to {session.sid}")
                    return
                except Exception:
                    continue

        logger.error("No working method to send typing indicator")

    async def _delayed_send_typing(self, session_obj: Session, delay: float):
        session = session_obj.sid
        try:
            await asyncio.sleep(delay)
            await self._send_typing(session_obj)
            if session not in self._loop_tasks or self._loop_tasks[session].done():
                self._typing_running[session] = True
                task = asyncio.create_task(self._typing_loop(session_obj))
                self._loop_tasks[session] = task
        except asyncio.CancelledError:
            logger.debug(f"Typing delayed task cancelled for {session}")

    async def _typing_loop(self, session_obj: Session):
        session = session_obj.sid
        while self._typing_running.get(session, False):
            try:
                await asyncio.sleep(self.typing_interval_seconds)
                if self._typing_running.get(session, False):
                    await self._send_typing(session_obj, log=False)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Typing loop error for {session}: {e}")
        logger.debug(f"Typing loop stopped for {session}")

    async def _typing_max_timeout(self, session_obj: Session, max_seconds: float):
        session = session_obj.sid
        try:
            await asyncio.sleep(max_seconds)
            self._stop_typing_loop(session_obj)
            logger.debug(f"Stopped typing for {session} due to max timeout ({max_seconds}s)")
        except asyncio.CancelledError:
            pass

    def _stop_typing_loop(self, session_obj: Session):
        session = session_obj.sid
        if session in self._typing_running:
            self._typing_running[session] = False
        if session in self._loop_tasks and not self._loop_tasks[session].done():
            self._loop_tasks[session].cancel()
        self._loop_tasks.pop(session, None)
        self._typing_running.pop(session, None)
        # 取消延迟任务，防止在sleep结束后启动新循环
        if session in self._delay_tasks and not self._delay_tasks[session].done():
            self._delay_tasks[session].cancel()
        self._delay_tasks.pop(session, None)
        # 取消最大时长兜底任务（若当前就在该任务内则只清理引用）
        max_task = self._max_tasks.pop(session, None)
        if max_task and not max_task.done() and max_task is not asyncio.current_task():
            max_task.cancel()

    def _start_typing_for_session(self, session_obj: Session):
        sid = session_obj.sid
        self._stop_typing_loop(session_obj)

        task = asyncio.create_task(self._delayed_send_typing(session_obj, self.typing_delay_seconds))
        self._delay_tasks[sid] = task
        task.add_done_callback(lambda t: self._delay_tasks.pop(sid, None))

        if self.typing_max_seconds > 0:
            max_task = asyncio.create_task(
                self._typing_max_timeout(session_obj, self.typing_max_seconds)
            )
            self._max_tasks[sid] = max_task
            max_task.add_done_callback(lambda t: self._max_tasks.pop(sid, None))

    # 在 llm_request 阶段启动（Priority.LOW，晚于限流等插件），避免未真正请求 LLM 时假输入中
    @on.llm_request(priority=Priority.LOW)
    async def handle_typing_indication(self, event: KiraMessageBatchEvent, req: LLMRequest, *_):
        if not self.typing_indicator_enabled:
            return
        if event.adapter.platform != "QQ":
            return
        # 只处理私聊
        if event.is_group_message():
            return
        if event.is_stopped:
            return
        self._start_typing_for_session(event.session)

    @on.llm_response(priority=Priority.HIGH)
    async def on_llm_response(self, event: KiraMessageBatchEvent, resp: LLMResponse):
        if not self.typing_indicator_enabled:
            return
        if event.adapter.platform != "QQ":
            return
        # 只处理私聊
        if event.is_group_message():
            return
        sid = event.sid
        if not resp.tool_calls:
            self._stop_typing_loop(event.session)
            logger.debug(f"Stopped typing loop for {sid} due to final response (no tool calls)")
