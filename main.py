import asyncio
import hashlib
import json
import re
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.message_components import At, Image, Reply
from astrbot.api.provider import Provider
from astrbot.api.star import Context, Star
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

try:
    from .renderer import AsciiArt, paint_ascii_art_png, render_ascii_art
except (
    ImportError
):  # Fallback when the plugin dir is imported as a plain module (tests).
    from renderer import AsciiArt, paint_ascii_art_png, render_ascii_art

EXTRACT_ATTEMPTS = 2
"""Scene extraction: 1 call + 1 retry on timeout/exception/illegal JSON."""
CACHE_MAX_ENTRIES = 200
IMAGE_RECALL_SECONDS = 600
CLAIM_TTL_SECONDS = 120

USAGE_TEXT = (
    "用法：发送 /拼豆 并附带一张图片，我会把图片拼成一幅彩色拼豆画。\n"
    "也可以：引用一张图片回复 /拼豆，或先发送图片、再在 10 分钟内发送 /拼豆。\n"
    "（当前消息未检测到图片）"
)
PROCESSING_TEXT = "正在埋头拼豆，请稍候…"
TRIGGER_REGEX = r"^\s*/\s*拼豆(?:\s|$)"
"""The one wake word: /拼豆 (slash required; not affected by wake_prefix)."""
"""Fallback trigger pattern; matches the command with or without a slash
prefix regardless of the host's wake_prefix configuration."""

# Substrings of provider errors that indicate a model/config problem rather
# than a transient failure; they make the degrade caption more specific.
_MODEL_CONFIG_ERROR_MARKERS = (
    "model_not_found",
    "no available channel",
    "invalid model",
    "invalid_model",
    "does not exist",
    "not support",
    "unsupported",
    "404",
)


def _extract_images(chain: list) -> list[Image]:
    """Collect image components from a message chain.

    Includes images quoted inside a Reply component (the user quoted an image
    message and sent the command as the reply).

    Args:
        chain: The message component chain.

    Returns:
        Images found in the chain, direct images first.
    """
    images = [c for c in chain if isinstance(c, Image)]
    for component in chain:
        if isinstance(component, Reply) and component.chain:
            images.extend(c for c in component.chain if isinstance(c, Image))
    return images


class PindoubaPlugin(Star):
    """Image-to-character-art plugin: pixel mapping renderer + VLM scene hints."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config

        plugin_dir = Path(__file__).parent
        try:
            self._prompts = {
                name: (plugin_dir / "prompts" / f"{name}.txt").read_text(
                    encoding="utf-8"
                )
                for name in ("extract_scene",)
            }
        except OSError as e:
            raise RuntimeError(f"[pindouba] failed to load plugin resource: {e}") from e

        # unified_msg_origin:sid -> last trigger timestamp
        self._last_trigger: dict[str, float] = {}
        # image content hash -> (timestamp, VLM scene params)
        self._cache: dict[str, tuple[float, dict]] = {}
        # message_id -> claim timestamp; dedupes the command/regex entry points
        self._inflight: dict[str, float] = {}
        # unified_msg_origin -> (timestamp, image ref) of the latest seen image
        self._last_image: dict[str, tuple[float, str]] = {}

    async def initialize(self):
        """Log readiness so the renderer is visible at startup."""
        logger.info("[pindouba] character-art renderer ready (Pillow pixel mapping)")

    # ------------------------------------------------------------------ #
    # Entry points
    # ------------------------------------------------------------------ #

    @filter.regex(TRIGGER_REGEX)
    async def pindouba_trigger(self, event: AstrMessageEvent):
        """The one true entry point: `/拼豆` (+ image), regardless of wake_prefix."""
        if not self._claim(event):
            return
        async for result in self._handle_trigger(event):
            yield result

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """Record the latest image per session; optional "@bot + image" trigger."""
        chain = event.get_messages()
        for component in chain:
            if isinstance(component, Image) and (component.url or component.file):
                self._remember_image(
                    event.unified_msg_origin, component.url or component.file
                )
                break

        if not self.config.get("allow_at_trigger", False):
            return

        text = (event.message_str or "").strip()
        if text.startswith("/") or "拼豆" in text:
            return

        self_id = str(event.get_self_id())
        at_me = any(isinstance(c, At) and str(c.qq) == self_id for c in chain)
        if not at_me or not any(isinstance(c, Image) for c in chain):
            return

        if not self._claim(event):
            return
        async for result in self._handle_trigger(event):
            yield result
        # The message is consumed here; keep the default LLM chat flow away.
        event.should_call_llm(False)
        event.stop_event()

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #

    async def _handle_trigger(
        self,
        event: AstrMessageEvent,
    ) -> AsyncGenerator[MessageEventResult, None]:
        """Shared body of every entry point: resolve the image, then generate.

        Args:
            event: The triggering message event.

        Yields:
            Message results: usage help, cooldown notice, the processing ack,
            or the rendered character art with its caption.
        """
        umo = event.unified_msg_origin
        direct = _extract_images(event.get_messages())
        notes: list[str] = []
        if len(direct) > 1:
            notes.append(f"检测到 {len(direct)} 张图片，仅处理第一张。")

        if direct:
            image = direct[0]
        else:
            # Fall back to the latest image seen in this session (e.g. the
            # user sent the picture in a separate message).
            recalled = self._recall_image(umo)
            if recalled is None:
                yield event.make_result().message(USAGE_TEXT)
                return
            image = Image(file=recalled)
            notes.append("已使用本会话最近发送的一张图片。")

        ok, notice = self._cooldown_gate(event)
        if not ok:
            if notice:
                yield event.make_result().message(notice)
            return

        async for result in self._generate_flow(event, image, notes):
            yield result

    async def _generate_flow(
        self,
        event: AstrMessageEvent,
        image: Image,
        notes: list[str],
    ) -> AsyncGenerator[MessageEventResult, None]:
        """Run the download -> (optional) scene extraction -> render pipeline.

        The image is downloaded first so the scene cache can be keyed by
        content hash (URLs change between forwards). The VLM is only called
        when a feature consumes its output (caption or crop box) and the
        cache misses; it merely supplies the subject crop box and caption,
        while the resemblance comes from the algorithmic renderer. Every
        model failure degrades to pure rendering instead of aborting.

        Args:
            event: The triggering message event.
            image: The image component to process.
            notes: User-facing notes to include in the caption.

        Yields:
            Message results: the processing ack, then the rendered art.
        """
        image_ref = image.url or image.file or ""
        if not image_ref:
            yield event.make_result().message(
                "图片数据异常（无可用的图片地址），请重新发送。"
            )
            return

        try:
            image_path = await image.convert_to_file_path()
            content_key = self._content_key(image_path)
        except Exception as e:
            logger.error(f"[pindouba] image download failed: {e}")
            yield event.make_result().message(
                "图片处理失败（下载失败或格式不支持），请换一张图片试试。"
            )
            return

        # The VLM is only worth calling when a feature actually consumes its
        # output (caption or crop box); otherwise render directly, no API cost.
        want_scene = bool(self.config.get("enable_caption", True)) or bool(
            self.config.get("auto_crop", False)
        )
        scene: dict | None = None
        degraded_reason = ""
        provider = None
        if want_scene:
            scene = self._cache_get(content_key)
            if scene is None:
                provider = self._resolve_provider(event)
                if provider is not None:
                    yield event.make_result().message(PROCESSING_TEXT)
                    try:
                        scene = await self._extract_scene(provider, image_ref)
                    except Exception as e:
                        logger.warning(
                            "[pindouba] scene extraction degraded to pure "
                            f"rendering: {e}"
                        )
                        degraded_reason = (
                            "模型不可用" if self._is_config_error(e) else "模型识别失败"
                        )
                    else:
                        self._cache_put(content_key, scene)

        try:
            art = self._render(image_path, scene)
        except ValueError as e:
            logger.warning(f"[pindouba] rendering rejected: {e}")
            yield event.make_result().message(
                "这张图不适合直接拼豆（尺寸过小或长宽比太悬殊，比如超长截图），"
                "裁剪一下或换一张试试。"
            )
            return
        except Exception as e:
            logger.error(f"[pindouba] rendering failed: {e}")
            yield event.make_result().message(
                "图片处理失败（下载失败或格式不支持），请换一张图片试试。"
            )
            return

        caption_bits = list(notes)
        if scene is not None:
            if scene.get("caption"):
                caption_bits.append(f"识别：{scene['caption']}")
        elif want_scene:
            if provider is None:
                caption_bits.append("未配置多模态模型，按原图直接渲染")
            elif degraded_reason:
                caption_bits.append(f"{degraded_reason}，已按原图直接渲染")

        if self.config.get("send_mode", "image") == "text" and art.kind != "bead":
            text = art.text
            if caption_bits:
                text += "\n\n" + "\n".join(caption_bits)
            yield event.make_result().message(text)
            return

        out_path = Path(get_astrbot_temp_path()) / (
            f"charart_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        )
        paint_ascii_art_png(art, out_path, color=bool(self.config.get("color", True)))
        yield event.make_result().file_image(str(out_path))
        if caption_bits:
            yield event.make_result().message("\n".join(caption_bits))

    async def _extract_scene(self, provider: Provider, image_ref: str) -> dict:
        """Ask the VLM for renderer parameters (subject crop box, caption).

        Args:
            provider: The chat provider (must support image input).
            image_ref: Image URL or local/base64 reference.

        Returns:
            The parsed scene dict; empty fields simply yield no hint.

        Raises:
            RuntimeError: If all attempts fail (call error or illegal JSON).
        """
        last_error = "unknown error"
        for _ in range(EXTRACT_ATTEMPTS):
            try:
                text = await self._call_llm(
                    provider, self._prompts["extract_scene"], image_urls=[image_ref]
                )
                data = self._parse_llm_json(text)
                if data:
                    return data
                last_error = f"illegal scene JSON: {text[:200]}"
            except Exception as e:
                last_error = str(e)
            logger.warning(f"[pindouba] scene extraction attempt failed: {last_error}")
        raise RuntimeError(
            f"scene extraction failed after {EXTRACT_ATTEMPTS} attempts: {last_error}"
        )

    def _render(self, image_path: str, scene: dict | None) -> AsciiArt:
        """Render the image file with parameters derived from config + scene.

        Args:
            image_path: Local path of the downloaded image.
            scene: VLM scene params (may be None for pure algorithmic mode).

        Returns:
            The rendered character grid.

        Raises:
            Exception: Propagated renderer errors (unreadable image, ...).
        """
        mode_cfg = str(self.config.get("render_mode", "auto"))
        # Bead mosaic reproduces the whole image (background included) and
        # keeps more fidelity than dot/char modes; classic modes stay as
        # explicit style choices.
        mode = "bead" if mode_cfg == "auto" else mode_cfg
        crop = self._scene_crop_fractions(scene)
        palette = bool(self.config.get("bead_palette", False))
        dither = bool(self.config.get("bead_dither", False))
        art = render_ascii_art(
            image_path,
            mode=mode,
            width=int(self.config.get("char_width", 80)),
            crop_box=crop,
            bead_palette=palette,
            bead_dither=dither,
        )
        logger.info(
            f"[pindouba] rendered {art.width}x{art.height} mode={mode} "
            f"cropped={crop is not None} palette={palette} dither={dither}"
        )
        return art

    def _scene_crop_fractions(
        self, scene: dict | None
    ) -> tuple[float, float, float, float] | None:
        """Extract a validated subject crop box from the scene params.

        Args:
            scene: VLM scene params (may be None).

        Returns:
            An (x1, y1, x2, y2) fraction box in [0, 1], or None when cropping
            is disabled / the box is missing, invalid, or covers the whole
            image.
        """
        if not self.config.get("auto_crop", False) or not scene:
            return None
        box = scene.get("subject_box")
        if not isinstance(box, list) or len(box) != 4:
            return None
        try:
            vals = [float(v) for v in box]
        except (TypeError, ValueError):
            return None
        if any(v < 0 for v in vals):
            return None
        peak = max(vals)
        if peak <= 1.0:
            fractions = vals
        elif peak <= 1000.0:  # 0-1000 normalized (qwen-vl grounding convention)
            fractions = [v / 1000 for v in vals]
        else:
            return None
        x1, y1, x2, y2 = fractions
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        if (x2 - x1) > 0.96 and (y2 - y1) > 0.96:
            return None  # subject fills the image, cropping is pointless
        if (x2 - x1) < 0.05 or (y2 - y1) < 0.05:
            return None  # too small to be a real subject box
        return (x1, y1, x2, y2)

    async def _call_llm(
        self,
        provider: Provider,
        prompt: str,
        image_urls: list[str] | None = None,
    ) -> str:
        """Call the provider with a hard timeout and return the completion text.

        Args:
            provider: The chat provider.
            prompt: The full prompt text.
            image_urls: Optional image references for vision input.

        Returns:
            The completion text (may be empty).

        Raises:
            TimeoutError: If the call exceeds call_timeout_seconds.
            Exception: Propagated provider errors (unsupported image input,
                network failures, ...).
        """
        timeout = float(self.config.get("call_timeout_seconds", 90))
        resp = await asyncio.wait_for(
            provider.text_chat(prompt=prompt, image_urls=image_urls),
            timeout=timeout,
        )
        return resp.completion_text or ""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_llm_json(text: str) -> dict | None:
        """Tolerantly parse a JSON object out of an LLM response.

        Strips markdown code fences and extracts the outermost `{...}` block.

        Args:
            text: Raw LLM completion text.

        Returns:
            The parsed dict, or None when no valid JSON object is found.
        """
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
            t = re.sub(r"\s*```$", "", t)
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(t[start : end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _is_config_error(error: Exception) -> bool:
        """Check whether a provider error looks like a model/config problem.

        Args:
            error: The caught exception from the LLM call.

        Returns:
            True for unknown-model / no-channel / unsupported-image errors.
        """
        text = str(error).lower()
        return any(marker in text for marker in _MODEL_CONFIG_ERROR_MARKERS)

    def _resolve_provider(self, event: AstrMessageEvent) -> Provider | None:
        """Resolve the chat provider from config or the session default.

        Args:
            event: The triggering message event.

        Returns:
            The provider, or None when unconfigured (pure algorithmic mode).
        """
        provider_id = str(self.config.get("provider_id") or "").strip()
        if provider_id:
            return self.context.get_provider_by_id(provider_id)
        return self.context.get_using_provider(event.unified_msg_origin)

    def _cooldown_gate(self, event: AstrMessageEvent) -> tuple[bool, str]:
        """Apply the per-user cooldown and arm it when passing.

        Keyed by session + sender so that in group chats one user's render
        does not put the whole group on cooldown.

        Args:
            event: The triggering message event.

        Returns:
            A (allowed, notice) tuple. When not allowed, `notice` is the
            cooldown message (or empty for silent mode).
        """
        key = f"{event.unified_msg_origin}:{event.get_sender_id()}"
        cooldown = int(self.config.get("cooldown_seconds", 30))
        now = time.time()
        elapsed = now - self._last_trigger.get(key, 0.0)
        if cooldown > 0 and elapsed < cooldown:
            remaining = int(cooldown - elapsed) + 1
            if self.config.get("cooldown_notice", True):
                return False, f"拼豆冷却中，请 {remaining} 秒后再试。"
            return False, ""
        self._last_trigger[key] = now
        return True, ""

    def _claim(self, event: AstrMessageEvent) -> bool:
        """Claim a message so only one entry point processes it.

        The command and regex handlers can both activate on the same message;
        whichever claims first wins.

        Args:
            event: The triggering message event.

        Returns:
            True when this call is the first claim for the message.
        """
        msg_id = str(event.message_obj.message_id)
        now = time.time()
        expired = [
            key for key, ts in self._inflight.items() if now - ts > CLAIM_TTL_SECONDS
        ]
        for key in expired:
            del self._inflight[key]
        if msg_id in self._inflight:
            return False
        self._inflight[msg_id] = now
        return True

    def _remember_image(self, umo: str, image_ref: str) -> None:
        """Remember the latest image reference seen in a session.

        Args:
            umo: The unified message origin identifying the session.
            image_ref: The image URL/file reference.
        """
        if len(self._last_image) >= CACHE_MAX_ENTRIES:
            oldest = min(self._last_image, key=lambda key: self._last_image[key][0])
            del self._last_image[oldest]
        self._last_image[umo] = (time.time(), image_ref)

    def _recall_image(self, umo: str) -> str | None:
        """Return the session's recent image reference, if still fresh.

        Args:
            umo: The unified message origin identifying the session.

        Returns:
            The image reference, or None when absent or expired.
        """
        entry = self._last_image.get(umo)
        if entry is None:
            return None
        ts, image_ref = entry
        if time.time() - ts > IMAGE_RECALL_SECONDS:
            del self._last_image[umo]
            return None
        return image_ref

    @staticmethod
    def _content_key(image_path: str) -> str:
        """Hash the image file bytes into a stable cache key.

        Chat platforms often re-serve the same picture under a fresh URL on
        every forward; keying the scene cache by content makes those hits.

        Args:
            image_path: Local path of the downloaded image.

        Returns:
            The MD5 hex digest of the file bytes.
        """
        return hashlib.md5(Path(image_path).read_bytes()).hexdigest()

    def _cache_get(self, image_hash: str) -> dict | None:
        """Look up a non-expired scene-params cache entry.

        Args:
            image_hash: The image content hash as cache key.

        Returns:
            The cached scene params, or None on miss/expiry/disabled cache.
        """
        if not self.config.get("enable_cache", True):
            return None
        entry = self._cache.get(image_hash)
        if entry is None:
            return None
        ts, scene = entry
        ttl = int(self.config.get("cache_ttl_minutes", 360)) * 60
        if time.time() - ts > ttl:
            del self._cache[image_hash]
            return None
        return scene

    def _cache_put(self, image_hash: str, scene: dict) -> None:
        """Store scene params in the bounded in-memory cache.

        Args:
            image_hash: The image content hash as cache key.
            scene: The VLM scene params to store.
        """
        if not self.config.get("enable_cache", True):
            return
        if len(self._cache) >= CACHE_MAX_ENTRIES:
            oldest = min(self._cache, key=lambda key: self._cache[key][0])
            del self._cache[oldest]
        self._cache[image_hash] = (time.time(), scene)
