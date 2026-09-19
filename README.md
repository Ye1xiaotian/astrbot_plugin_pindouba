# 拼豆吧

AstrBot 插件。把一张图片变成像拼豆一样

不配模型也能用。配置一个支持图片输入的模型后，还能裁掉背景、识别图片内容；模型不在或调用失败时，插件照常出图

## 怎么用

引用图片发送 `/拼豆`，
拼头像发 `/拼我`，
`/拼@群友` ，可拼群友头像。
私聊群聊都一样，不需要机器人的唤醒前缀

发图有三种方式：

| 方式 | 操作 |
|------|------|
| 同一条消息 | 发送 `/拼豆` 并附带一张图片 |
| 引用回复 | 引用一张图片消息，回复 `/拼豆` |
| 先图后令 | 先发图片，10 分钟内补发 `/拼豆` |

一次发了多张图，只拼第一张

拼头像不用发图（仅支持 QQ）：

| 命令 | 效果 |
|------|------|
| `/拼我` | 拼你自己的头像 |
| `/拼 @某人` | 拼对方的头像，群里拿来互相拼最合适 |

同一张图再拼一次（就算链接变了）会跳过识别直接出图

冷却按人算：同一个人 30 秒内拼不了第二次，群里别人不受影响。时长可在配置里调

## 功能说明

- **拼豆画**（默认）：整张图铺成彩色方格，像放大后的像素图。把 `char_width` 调小就更粗犷，调大就更细腻（可根据自己喜好自行调整）
- **换画法**：把 `render_mode` 改成 `braille`（密集小点）、`shading`（黑白深浅）或 `line`（只勾轮廓线）
- **拼豆色卡**：打开 `bead_palette` 后，颜色只从 56 色经典拼豆色卡里选，更有手工感，默认关闭（测试下来若原图色彩过于丰富，会与原图色差较大，一般不推荐开启）。选色按人眼对绿色更敏感的规则加权，天空和肤色会更准一些
- **色板抖动**：`bead_palette` 开着时再开 `bead_dither`，相邻格子互相匀色（误差扩散），渐变过渡顺滑、不容易出大块色带，近看会有细颗粒感。色卡关着时这项没作用。
- **只拼主体**：打开 `auto_crop`，自动裁掉背景。需要模型支持，默认关闭（识别内容和多模态模型能力相关）
- **识别说明**：配了模型时，成品下面会附一句"识别：xxx"。不需要的话关掉 `enable_caption`，再关掉 `auto_crop`，就完全不调用模型，出图最快、不耗 API。
- **长图**：长宽比超过 1:6 的图（比如整页聊天截图）拼出来会细成一条，插件会提醒你裁剪后再发，不硬拼
- **拼头像**：`/拼我` 拼自己的头像，`/拼 @某人` 拼群友的头像。仅支持 QQ，走 QQ 公开的头像直链，不调用模型，发完基本秒回
- **@ 触发**：打开 `allow_at_trigger` 后，@机器人并发图片也能拼，不用打命令

## 四种画法对比

同一张图、宽度都是 80 时的效果：

| 拼豆（默认） | 盲文小点（`braille`） |
|---|---|
<!-- Images use absolute raw.githubusercontent URLs: the dashboard renders this
     README at its own origin, where relative asset paths 404. -->
| <img src="https://raw.githubusercontent.com/Ye1xiaotian/astrbot_plugin_pindouba/main/assets/compare_bead.png" width="340"> | <img src="https://raw.githubusercontent.com/Ye1xiaotian/astrbot_plugin_pindouba/main/assets/compare_braille.png" width="340"> |
| 彩色方格铺满整图 | 黑白小点最密|

| 灰阶（`shading`） | 描线（`line`） |
|---|---|
| <img src="https://raw.githubusercontent.com/Ye1xiaotian/astrbot_plugin_pindouba/main/assets/compare_shading.png" width="340"> | <img src="https://raw.githubusercontent.com/Ye1xiaotian/astrbot_plugin_pindouba/main/assets/compare_line.png" width="340"> |
| 黑白深浅过渡| 只勾轮廓，适合线稿 |

喜欢哪种就改 `render_mode`。（除bead模式外，其他模式均为迭代产物，不推荐使用）

## 配置项

| 配置 | 默认 | 说明 |
|------|------|------|
| `provider_id` | 空 | 支持多模态的模型 ID。留空则用当前聊天的默认模型 |
| `render_mode` | auto | auto/bead 为拼豆画；braille 密集小点；shading 黑白深浅；line 轮廓线 |
| `char_width` | 80 | 画的宽度（列数），16–120 | 
| `auto_crop` | false | 裁掉背景只拼主体（需要模型） |
| `enable_caption` | true | 成品下附一句"识别：xxx"；和 `auto_crop` 都关掉后跳过模型，出图最快 |
| `bead_palette` | false | 用 56 色拼豆色卡选色 |
| `bead_dither` | false | 色板抖动：渐变过渡更顺滑，仅 `bead_palette` 开启时生效 |
| `color` | true | 彩色显示，关闭则单色 |
| `send_mode` | image | image 渲染成图片发送（推荐）；text 直接发字符文本，仅对 braille/shading/line 生效，bead 仍发图片 |
| `call_timeout_seconds` | 90 | 等模型回复的最长时间（秒） |
| `allow_at_trigger` | false | 允许 @机器人 + 图片触发 |
| `cooldown_seconds` | 30 | 同一个人在同一聊天的冷却秒数，群里互不影响，0 为不冷却 |
| `cooldown_notice` | true | 冷却时是否回复提示 |
| `enable_cache` | true | 同一张图不重复识别（链接变了也算同一张） |
| `cache_ttl_minutes` | 360 | 识别记录保留多久（分钟） |

## 隐私

配了模型时，你发的图片会送到**你自己配置的那个模型**那里识别，别处不去。不配模型，图片只在你自己的机器上处理。拼头像只用到 QQ 公开的头像直链，不涉及其他信息

## 开发

```
放进 data/plugins/ 重载即可
python tests/test_plugin_logic.py   # 插件逻辑测试（无需 astrbot 环境）
python tests/test_renderer.py       # 渲染测试（需要 Pillow）
```

渲染器在 `renderer.py` 一个文件里：加画法、改豆形、调色板可以在该文件里修改

## License

MIT
