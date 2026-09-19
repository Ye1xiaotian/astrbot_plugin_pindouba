"""Self-contained logic tests for astrbot_plugin_kaomoji (character-art plugin).

Runs with a plain Python interpreter (no astrbot install required): when the
real `astrbot` package is unavailable, minimal stubs are injected first.
Pillow is required for the pipeline tests (renderer dependency).

Run: python tests/test_plugin_logic.py
"""

import asyncio
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PLUGIN_DIR))

# Filled by the stub filter below; stays empty when the real astrbot is used.
REGISTERED_COMMANDS: list[str] = []

try:
    import main  # noqa: F401  (real astrbot available)
except ImportError:
    # Stub the astrbot plugin API surface used by main.py.
    import types

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")

    class _Logger:
        def __getattr__(self, name):
            return lambda *a, **k: None

    api.logger = _Logger()

    class AstrBotConfig(dict):
        pass

    api.AstrBotConfig = AstrBotConfig

    event_mod = types.ModuleType("astrbot.api.event")

    class AstrMessageEvent:
        pass

    class MessageEventResult:
        pass

    class _Filter:
        EventMessageType = type("EventMessageType", (), {"ALL": "ALL"})

        def command(self, name, *a, **k):
            REGISTERED_COMMANDS.append(name)
            return lambda f: f

        def event_message_type(self, *a, **k):
            return lambda f: f

    event_mod.AstrMessageEvent = AstrMessageEvent
    event_mod.MessageEventResult = MessageEventResult
    event_mod.filter = _Filter()

    msgcomp = types.ModuleType("astrbot.api.message_components")

    class At:
        pass

    class Image:
        # Mirrors the real component signature: `file` is a required
        # positional argument, so url-only constructions fail in tests too.
        def __init__(self, file, url="", **_):
            self.url = url
            self.file = file

        @staticmethod
        def fromURL(url):
            return Image(file=url)

    class Reply:
        def __init__(self, chain=None):
            self.chain = chain or []

    msgcomp.At = At
    msgcomp.Image = Image
    msgcomp.Reply = Reply

    provider_mod = types.ModuleType("astrbot.api.provider")
    provider_mod.Provider = type("Provider", (), {})

    star_mod = types.ModuleType("astrbot.api.star")

    class Context:
        pass

    class Star:
        def __init__(self, context):
            self.context = context

    star_mod.Context = Context
    star_mod.Star = Star

    def _get_astrbot_temp_path():
        return tempfile.gettempdir()

    path_mod = types.ModuleType("astrbot.core.utils.astrbot_path")
    path_mod.get_astrbot_temp_path = _get_astrbot_temp_path

    for name, mod in {
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.event": event_mod,
        "astrbot.api.message_components": msgcomp,
        "astrbot.api.provider": provider_mod,
        "astrbot.api.star": star_mod,
        "astrbot.core": types.ModuleType("astrbot.core"),
        "astrbot.core.utils": types.ModuleType("astrbot.core.utils"),
        "astrbot.core.utils.astrbot_path": path_mod,
    }.items():
        sys.modules[name] = mod
    import main


def make_plugin(config: dict) -> main.PindoubaPlugin:
    """Build a plugin instance with real resources and no Star machinery."""
    plugin = main.PindoubaPlugin.__new__(main.PindoubaPlugin)
    plugin._prompts = {
        "extract_scene": (PLUGIN_DIR / "prompts" / "extract_scene.txt").read_text(
            encoding="utf-8"
        )
    }
    plugin.config = dict(config)
    plugin._cache = {}
    plugin._last_trigger = {}
    plugin._inflight = {}
    plugin._last_image = {}
    return plugin


DEFAULT_CONFIG = {
    "provider_id": "",
    "call_timeout_seconds": 90,
    "render_mode": "auto",
    "char_width": 40,
    "bead_palette": False,
    "bead_dither": False,
    "auto_crop": True,
    "enable_caption": True,
    "color": True,
    "send_mode": "image",
    "allow_at_trigger": False,
    "cooldown_seconds": 30,
    "cooldown_notice": True,
    "enable_cache": True,
    "cache_ttl_minutes": 360,
}


class FakeResult:
    """Records a chained message/file_image call."""

    def __init__(self):
        self.text = None
        self.image_path = None

    def message(self, text):
        self.text = text
        return self

    def file_image(self, path):
        self.image_path = path
        return self


class FakeImage:
    """Image component backed by a real temp PNG (renderer needs pixels)."""

    _png_path = None

    def __init__(self, url=""):
        self.url = url
        self.file = ""

    @classmethod
    async def convert_to_file_path(cls):
        if cls._png_path is None:
            from PIL import Image

            img = Image.new("L", (100, 50))
            for x in range(100):
                for y in range(50):
                    img.putpixel((x, y), int(x / 99 * 255))
            cls._png_path = str(Path(tempfile.mkdtemp()) / "fake.png")
            img.convert("RGB").save(cls._png_path)
        return cls._png_path


class FakeTallImage:
    """Image component backed by a very tall PNG (aspect guard test)."""

    _png_path = None

    def __init__(self, url="http://x/tall.jpg"):
        self.url = url
        self.file = ""

    @classmethod
    async def convert_to_file_path(cls):
        if cls._png_path is None:
            from PIL import Image

            img = Image.new("L", (40, 400), 128)
            cls._png_path = str(Path(tempfile.mkdtemp()) / "tall.png")
            img.convert("RGB").save(cls._png_path)
        return cls._png_path


class FakeEvent:
    unified_msg_origin = "p:g:s"
    _id_seq = itertools.count()

    def __init__(self):
        self.results = []
        self.sender_id = "u1"
        self.platform_name = "aiocqhttp"
        self.chain = []
        self.message_obj = type(
            "M", (), {"message_id": f"m{next(FakeEvent._id_seq)}"}
        )()

    def make_result(self):
        result = FakeResult()
        self.results.append(result)
        return result

    def get_messages(self):
        return self.chain

    def get_sender_id(self):
        return self.sender_id

    def get_platform_name(self):
        return self.platform_name


class TriggerTests(unittest.TestCase):
    def setUp(self):
        self.plugin = make_plugin(DEFAULT_CONFIG)

    def test_commands_registered_and_distinct(self):
        # The dashboard lists filter.command names and the host CommandFilter
        # derives mutual exclusion from them, so 拼豆/拼我/拼 must register
        # exactly once each.
        if not REGISTERED_COMMANDS:
            self.skipTest("filter stub not in use (real astrbot)")
        self.assertEqual(sorted(REGISTERED_COMMANDS), ["拼", "拼我", "拼豆"])

    def test_claim_dedupes_message(self):
        event = type("E", (), {})()
        event.message_obj = type("M", (), {"message_id": "m1"})()
        self.assertTrue(self.plugin._claim(event))
        self.assertFalse(self.plugin._claim(event))
        event.message_obj.message_id = "m2"
        self.assertTrue(self.plugin._claim(event))

    def test_remember_and_recall_image(self):
        self.plugin._remember_image("umo", "http://x/1.jpg")
        self.assertEqual(self.plugin._recall_image("umo"), "http://x/1.jpg")
        ts, ref = self.plugin._last_image["umo"]
        self.plugin._last_image["umo"] = (ts - main.IMAGE_RECALL_SECONDS - 1, ref)
        self.assertIsNone(self.plugin._recall_image("umo"))

    def test_extract_images_direct_and_reply(self):
        direct = main.Image(file="", url="http://a/1.jpg")
        quoted = main.Image(file="base64://x")
        self.assertEqual(main._extract_images([direct]), [direct])
        reply = main.Reply(chain=[quoted])
        self.assertEqual(main._extract_images([reply, direct]), [direct, quoted])
        self.assertEqual(main._extract_images([main.Reply(chain=[])]), [])


class HelperTests(unittest.TestCase):
    def setUp(self):
        self.plugin = make_plugin(DEFAULT_CONFIG)

    def test_parse_llm_json_tolerates_fences_and_noise(self):
        self.assertEqual(
            self.plugin._parse_llm_json('```json\n{"a": 1}\n```'), {"a": 1}
        )
        self.assertEqual(
            self.plugin._parse_llm_json('noise {"caption": "猫"} tail'),
            {"caption": "猫"},
        )
        self.assertIsNone(self.plugin._parse_llm_json("no json here"))

    def test_cooldown_gate(self):
        ok, notice = self.plugin._cooldown_gate(FakeEvent())
        self.assertTrue(ok)
        ok2, notice2 = self.plugin._cooldown_gate(FakeEvent())
        self.assertFalse(ok2)
        self.assertIn("秒", notice2)

    def test_cooldown_silent_mode(self):
        self.plugin.config["cooldown_notice"] = False
        self.plugin._cooldown_gate(FakeEvent())
        ok, notice = self.plugin._cooldown_gate(FakeEvent())
        self.assertFalse(ok)
        self.assertEqual(notice, "")

    def test_cooldown_independent_per_sender(self):
        e1, e2 = FakeEvent(), FakeEvent()
        e2.sender_id = "u2"
        self.assertTrue(self.plugin._cooldown_gate(e1)[0])
        self.assertTrue(self.plugin._cooldown_gate(e2)[0])  # other user passes
        self.assertFalse(self.plugin._cooldown_gate(e1)[0])  # same user blocked

    def test_content_key_stable_and_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "a.png"
            p2 = Path(tmp) / "b.png"
            p1.write_bytes(b"same")
            p2.write_bytes(b"other")
            key = main.PindoubaPlugin._content_key
            self.assertEqual(key(str(p1)), key(str(p1)))
            self.assertNotEqual(key(str(p1)), key(str(p2)))

    def test_cache_roundtrip(self):
        self.plugin._cache_put("ref", {"caption": "猫"})
        self.assertEqual(self.plugin._cache_get("ref"), {"caption": "猫"})
        self.assertIsNone(self.plugin._cache_get("missing"))

    def test_scene_crop_fractions(self):
        self.assertEqual(
            self.plugin._scene_crop_fractions({"subject_box": [100, 100, 900, 900]}),
            (0.1, 0.1, 0.9, 0.9),
        )
        # fractional box
        self.assertEqual(
            self.plugin._scene_crop_fractions({"subject_box": [0.1, 0.1, 0.9, 0.9]}),
            (0.1, 0.1, 0.9, 0.9),
        )
        # full image -> pointless crop
        self.assertIsNone(
            self.plugin._scene_crop_fractions({"subject_box": [0, 0, 1000, 1000]})
        )
        # invalid inputs
        self.assertIsNone(self.plugin._scene_crop_fractions({"subject_box": None}))
        self.assertIsNone(self.plugin._scene_crop_fractions({"subject_box": [1, 2]}))
        self.assertIsNone(
            self.plugin._scene_crop_fractions({"subject_box": [1, 2, 3, 9999]})
        )
        # disabled via config
        self.plugin.config["auto_crop"] = False
        self.assertIsNone(
            self.plugin._scene_crop_fractions({"subject_box": [100, 100, 900, 900]})
        )

    def test_is_config_error(self):
        self.assertTrue(
            main.PindoubaPlugin._is_config_error(RuntimeError("503 model_not_found: x"))
        )
        self.assertFalse(
            main.PindoubaPlugin._is_config_error(RuntimeError("connection reset"))
        )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.plugin = make_plugin(DEFAULT_CONFIG)

    def _wire_provider(self, provider):
        self.plugin.context = type("Ctx", (), {})()
        self.plugin.context.get_using_provider = lambda umo=None: provider
        self.plugin.context.get_provider_by_id = lambda pid=None: None

    def test_extract_scene_retries_then_raises(self):
        calls = {"n": 0}

        class BadProvider:
            async def text_chat(self, **kw):
                calls["n"] += 1
                raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            asyncio.run(self.plugin._extract_scene(BadProvider(), "http://x/img.jpg"))
        self.assertEqual(calls["n"], main.EXTRACT_ATTEMPTS)

    def test_generate_flow_without_provider_pure_renders(self):
        self.plugin.context = type("Ctx", (), {})()
        self.plugin.context.get_using_provider = lambda umo=None: None
        self.plugin.context.get_provider_by_id = lambda pid=None: None

        async def run():
            event = FakeEvent()
            flows = [
                r
                async for r in self.plugin._generate_flow(
                    event, FakeImage("http://x/1.jpg"), []
                )
            ]
            return flows

        flows = asyncio.run(run())
        self.assertEqual(len(flows), 2)
        self.assertTrue(flows[0].image_path)
        self.assertTrue(flows[0].image_path.endswith(".png"))
        self.assertIn("未配置多模态模型", flows[1].text)

    def test_generate_flow_with_scene_params_and_cache(self):
        scene_calls = {"n": 0}

        class SceneProvider:
            async def text_chat(self, **kw):
                scene_calls["n"] += 1
                return FakeResp(
                    '{"image_style": "lineart", "subject_box": [100, 100, 900, 900], '
                    '"caption": "测试猫"}'
                )

        self._wire_provider(SceneProvider())

        async def run():
            event = FakeEvent()
            flows = [
                r
                async for r in self.plugin._generate_flow(
                    event, FakeImage("http://x/cat.jpg"), []
                )
            ]
            return event, flows

        event, flows = asyncio.run(run())
        # ack + image + caption
        self.assertEqual(flows[0].text, main.PROCESSING_TEXT)
        self.assertTrue(flows[1].image_path)
        self.assertIn("识别：测试猫", flows[2].text)
        self.assertEqual(scene_calls["n"], 1)
        # second run hits the scene cache: no new model call, no ack
        event2, flows2 = asyncio.run(run())
        self.assertEqual(scene_calls["n"], 1)
        self.assertEqual(len(flows2), 2)
        self.assertIn("识别：测试猫", flows2[1].text)

    def test_generate_flow_degrades_on_model_error(self):
        class BrokenProvider:
            async def text_chat(self, **kw):
                raise RuntimeError("503 model_not_found")

        self._wire_provider(BrokenProvider())

        async def run():
            event = FakeEvent()
            return [
                r
                async for r in self.plugin._generate_flow(
                    event, FakeImage("http://x/cat.jpg"), []
                )
            ]

        flows = asyncio.run(run())
        self.assertEqual(flows[0].text, main.PROCESSING_TEXT)
        self.assertTrue(flows[1].image_path)  # art is still rendered
        self.assertIn("模型不可用，已按原图直接渲染", flows[2].text)

    def test_generate_flow_text_mode(self):
        self.plugin.config["send_mode"] = "text"
        self.plugin.config["render_mode"] = "shading"  # ascii kinds have text form
        self.plugin.context = type("Ctx", (), {})()
        self.plugin.context.get_using_provider = lambda umo=None: None
        self.plugin.context.get_provider_by_id = lambda pid=None: None

        async def run():
            event = FakeEvent()
            return [
                r
                async for r in self.plugin._generate_flow(
                    event, FakeImage("http://x/cat.jpg"), []
                )
            ]

        flows = asyncio.run(run())
        self.assertIsNone(flows[0].image_path)
        self.assertIn("\n", flows[0].text)  # multi-line art text

    def test_generate_flow_skips_vlm_when_features_off(self):
        self.plugin.config["enable_caption"] = False
        self.plugin.config["auto_crop"] = False
        calls = {"n": 0}

        class CountingProvider:
            async def text_chat(self, **kw):
                calls["n"] += 1
                return FakeResp("{}")

        self._wire_provider(CountingProvider())

        async def run():
            event = FakeEvent()
            return [
                r
                async for r in self.plugin._generate_flow(
                    event, FakeImage("http://x/a.jpg"), []
                )
            ]

        flows = asyncio.run(run())
        self.assertEqual(calls["n"], 0)  # model never called
        self.assertEqual(len(flows), 1)  # image only: no ack, no caption
        self.assertTrue(flows[0].image_path)

    def test_scene_cache_hits_across_url_change(self):
        scene_calls = {"n": 0}

        class SceneProvider:
            async def text_chat(self, **kw):
                scene_calls["n"] += 1
                return FakeResp('{"subject_box": null, "caption": "测试猫"}')

        self._wire_provider(SceneProvider())

        async def run(ref):
            event = FakeEvent()
            return [
                r async for r in self.plugin._generate_flow(event, FakeImage(ref), [])
            ]

        flows = asyncio.run(run("http://x/a.jpg"))
        self.assertEqual(len(flows), 3)
        # Same picture bytes under a fresh URL: cache hit, no second call.
        flows = asyncio.run(run("http://x/b.jpg"))
        self.assertEqual(scene_calls["n"], 1)
        self.assertEqual(len(flows), 2)
        self.assertIn("识别：测试猫", flows[1].text)

    def test_generate_flow_rejects_elongated_image(self):
        self.plugin.context = type("Ctx", (), {})()
        self.plugin.context.get_using_provider = lambda umo=None: None
        self.plugin.context.get_provider_by_id = lambda pid=None: None

        async def run():
            event = FakeEvent()
            return [
                r async for r in self.plugin._generate_flow(event, FakeTallImage(), [])
            ]

        flows = asyncio.run(run())
        self.assertEqual(len(flows), 1)
        self.assertIn("裁剪", flows[0].text)

    def test_generate_flow_no_image_ref(self):
        self.plugin.context = type("Ctx", (), {})()
        self.plugin.context.get_using_provider = lambda umo=None: None
        self.plugin.context.get_provider_by_id = lambda pid=None: None

        async def run():
            event = FakeEvent()
            return [
                r async for r in self.plugin._generate_flow(event, FakeImage(""), [])
            ]

        flows = asyncio.run(run())
        self.assertIn("图片数据异常", flows[0].text)


class AvatarStubImage(main.Image):
    """Image stub that records the constructed URL and serves real pixels."""

    last_url = ""
    _png_path = None

    def __init__(self, file, url="", **_):
        super().__init__(file=file, url=url)
        AvatarStubImage.last_url = url or file

    @classmethod
    def fromURL(cls, url):
        return cls(file=url)

    @classmethod
    async def convert_to_file_path(cls):
        if cls._png_path is None:
            from PIL import Image

            img = Image.new("L", (64, 64), 128)
            cls._png_path = str(Path(tempfile.mkdtemp()) / "avatar.png")
            img.convert("RGB").save(cls._png_path)
        return cls._png_path


class AvatarTests(unittest.TestCase):
    def setUp(self):
        self.plugin = make_plugin(DEFAULT_CONFIG)
        self._orig_image = main.Image
        main.Image = AvatarStubImage
        self.addCleanup(setattr, main, "Image", self._orig_image)
        AvatarStubImage.last_url = ""

    def _drain(self, agen):
        async def collect():
            return [r async for r in agen]

        return asyncio.run(collect())

    def test_avatar_refuses_non_qq_platform(self):
        event = FakeEvent()
        event.platform_name = "telegram"
        flows = self._drain(self.plugin._handle_avatar(event, None, "已使用你的头像。"))
        self.assertIn("只支持 QQ", flows[0].text)

    def test_avatar_self_uses_sender_qq(self):
        flows = self._drain(
            self.plugin._handle_avatar(FakeEvent(), None, "已使用你的头像。")
        )
        self.assertIn("nk=u1", AvatarStubImage.last_url)
        self.assertTrue(flows[0].image_path)
        self.assertIn("已使用你的头像", flows[1].text)

    def test_avatar_at_trigger_uses_at_target(self):
        at = main.At()
        at.qq = "12345"
        at.name = "小明"
        event = FakeEvent()
        event.chain = [at]
        flows = self._drain(self.plugin.avatar_at_trigger(event))
        self.assertIn("nk=12345", AvatarStubImage.last_url)
        self.assertIn("@小明", flows[1].text)

    def test_avatar_at_trigger_unnamed_target(self):
        at = main.At()
        at.qq = 67890  # adapters may hand over an int
        at.name = None
        event = FakeEvent()
        event.chain = [at]
        flows = self._drain(self.plugin.avatar_at_trigger(event))
        self.assertIn("nk=67890", AvatarStubImage.last_url)
        self.assertIn("已使用对方的头像", flows[1].text)

    def test_avatar_at_trigger_needs_a_target(self):
        flows = self._drain(self.plugin.avatar_at_trigger(FakeEvent()))
        self.assertIn("用法", flows[0].text)
        at = main.At()
        at.qq = "all"  # @全体成员 has no avatar to bead
        at.name = "@全体成员"
        event = FakeEvent()
        event.chain = [at]
        flows = self._drain(self.plugin.avatar_at_trigger(event))
        self.assertIn("用法", flows[0].text)

    def test_avatar_skips_vlm_even_when_enabled(self):
        calls = {"n": 0}

        class CountingProvider:
            async def text_chat(self, **kw):
                calls["n"] += 1
                return FakeResp("{}")

        self.plugin.context = type("Ctx", (), {})()
        self.plugin.context.get_using_provider = lambda umo=None: CountingProvider()
        self.plugin.context.get_provider_by_id = lambda pid=None: None
        # DEFAULT_CONFIG has enable_caption=True; avatar mode must override it.
        flows = self._drain(
            self.plugin._handle_avatar(FakeEvent(), None, "已使用你的头像。")
        )
        self.assertEqual(calls["n"], 0)
        self.assertTrue(flows[0].image_path)

    def test_avatar_is_cooldown_gated(self):
        self._drain(self.plugin._handle_avatar(FakeEvent(), None, "已使用你的头像。"))
        flows = self._drain(
            self.plugin._handle_avatar(FakeEvent(), None, "已使用你的头像。")
        )
        self.assertIn("冷却", flows[0].text)


class FakeResp:
    """Mimics the LLMResponse attribute surface used by main.py."""

    def __init__(self, text: str):
        self.completion_text = text


if __name__ == "__main__":
    unittest.main(verbosity=2)
