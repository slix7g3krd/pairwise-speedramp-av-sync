# pairwise-speedramp-av-sync

## 领域

游戏、图形与多媒体

## 主题

变速剪辑音画同步

## 说明

边界映射连续、分段变速与时间线排序。

初始状态下 `media/speedramp_av_sync.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
